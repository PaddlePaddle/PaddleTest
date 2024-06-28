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



class PrimitiveOp_1bf6bad4510211d62745017b64367c7b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.flatten(input_0, 2, 3), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_e18368ca924a14d069a08adacbdbb1e1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1bf6bad4510211d62745017b64367c7b
    def get_inputs(self):
        return [
            paddle.uniform([1, 15, 32, 32], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_068cad283dde9c2a4f3e277140b32fce(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1bf6bad4510211d62745017b64367c7b
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 32, 32], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ac9cc2ca95289bfcac2a209e0e0a55ee(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1bf6bad4510211d62745017b64367c7b
    def get_inputs(self):
        return [
            paddle.uniform([1, 91, 32, 32], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_8e6ad182ad0c495c1c780bc9fe29ab67(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1bf6bad4510211d62745017b64367c7b
    def get_inputs(self):
        return [
            paddle.uniform([16, 128, 16, 64], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_4be34f96ca4d3836f254ff4cb1e59fb1(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.flatten(input_0, 1, 3), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_7ee0d7aec82a174cb94d87708e0d9bae(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4be34f96ca4d3836f254ff4cb1e59fb1
    def get_inputs(self):
        return [
            paddle.uniform([512, 256, 7, 7], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_8cfdbbc6c3258c4b7f2d2b93d77d40d4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1bf6bad4510211d62745017b64367c7b
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 84, 84], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c42abbc8cf618e198bcdce62780750f9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1bf6bad4510211d62745017b64367c7b
    def get_inputs(self):
        return [
            paddle.uniform([1, 68, 84, 84], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_29bd7ae6da976a51d017427f120e47b2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1bf6bad4510211d62745017b64367c7b
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 16, 16], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_bd57eeb66282d00db1b9063207346203(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1bf6bad4510211d62745017b64367c7b
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 16, 16], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f521d676ab5f5dca6b973c28c7c4322c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1bf6bad4510211d62745017b64367c7b
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 16, 16], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_406dfd80a1fe11fbf2de3d1e327d8022(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1bf6bad4510211d62745017b64367c7b
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 24, 24], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_8f4cb3efe58db72935adf1f5ba125bdc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1bf6bad4510211d62745017b64367c7b
    def get_inputs(self):
        return [
            paddle.uniform([1, 68, 24, 24], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_7cb6c9421e94e952e13529aad61de002(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1bf6bad4510211d62745017b64367c7b
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 48, 48], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_75d6a308f9dd5bd740e50db3f34ec3cf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1bf6bad4510211d62745017b64367c7b
    def get_inputs(self):
        return [
            paddle.uniform([1, 68, 48, 48], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_70c246f46bbc33b70cdb86c88dcbce2c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1bf6bad4510211d62745017b64367c7b
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 15, 15], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_12e04b2e478feec2f8d83d54de597097(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1bf6bad4510211d62745017b64367c7b
    def get_inputs(self):
        return [
            paddle.uniform([1, 68, 15, 15], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_a71b4b0ddcf6f0f537f53ec17c571edc(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.flatten(input_0, 0, 2), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_8c49f1b9adaff854ab258e310e186c2a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a71b4b0ddcf6f0f537f53ec17c571edc
    def get_inputs(self):
        return [
            paddle.to_tensor([[[3]]], dtype='int32').reshape([1, 1, 1]),
        ]


class PrimitiveOp_eafb95a73f2f5fc792a56e4f0b174a0b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.flatten(input_0, 0, 1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_ed2042703495471f31e29db8ffc5d7ad(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_eafb95a73f2f5fc792a56e4f0b174a0b
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1, 2100], dtype='int64'),
        ]


class TestPrimitiveOp_eb892844b798df7ff0cd065c1652564a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1bf6bad4510211d62745017b64367c7b
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 64, 64], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9d2039c2fe8a5d990412c8bcc251c5c5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1bf6bad4510211d62745017b64367c7b
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 64, 64], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_bbc48e5d149e0e6357f7751eaef8d7cf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1bf6bad4510211d62745017b64367c7b
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 64, 64], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ef55dd202641dc4c1cdb08f80699ca0b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1bf6bad4510211d62745017b64367c7b
    def get_inputs(self):
        return [
            paddle.uniform([128, 320, 8, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f8a52a7e27848d2d161edb5e3379008c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1bf6bad4510211d62745017b64367c7b
    def get_inputs(self):
        return [
            paddle.uniform([1, 15, 64, 64], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9d2039c2fe8a5d990412c8bcc251c5c5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1bf6bad4510211d62745017b64367c7b
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 64, 64], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_833ca5a7bd86e0f68d1d8e075b27100b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1bf6bad4510211d62745017b64367c7b
    def get_inputs(self):
        return [
            paddle.uniform([1, 91, 64, 64], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ae1e4c190fa4b2bb0525f8bf8fec007f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1bf6bad4510211d62745017b64367c7b
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 26, 26], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_da6407e0865f0eea3ee863c0061764d8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1bf6bad4510211d62745017b64367c7b
    def get_inputs(self):
        return [
            paddle.uniform([1, 76, 26, 26], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_15877641890b0b332a8d040045418eee(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4be34f96ca4d3836f254ff4cb1e59fb1
    def get_inputs(self):
        return [
            paddle.uniform([11, 2048, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9ae8b8232489b323a3692a3f1de95f80(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4be34f96ca4d3836f254ff4cb1e59fb1
    def get_inputs(self):
        return [
            paddle.uniform([10, 1000, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_72c3ee56c11693f10b06c004ab96b2ab(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1bf6bad4510211d62745017b64367c7b
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 30, 30], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_fb4a938c44a69e3bf11577096cf4baac(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1bf6bad4510211d62745017b64367c7b
    def get_inputs(self):
        return [
            paddle.uniform([1, 68, 30, 30], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_edf2d260a393a90d18ab5841f7befc72(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1bf6bad4510211d62745017b64367c7b
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 52, 52], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_bff9935367f55f43614733531941911b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1bf6bad4510211d62745017b64367c7b
    def get_inputs(self):
        return [
            paddle.uniform([1, 68, 52, 52], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_80ec5596ccbf7204072a901887705ee6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1bf6bad4510211d62745017b64367c7b
    def get_inputs(self):
        return [
            paddle.uniform([1, 15, 8, 8], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_76362b8d422ad609daf0ec1a2359b77c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1bf6bad4510211d62745017b64367c7b
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 13, 13], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c2a43d9c1ceea934864401c873f26e5b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1bf6bad4510211d62745017b64367c7b
    def get_inputs(self):
        return [
            paddle.uniform([1, 76, 13, 13], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_55b8aaa0e0453ee8b26e21e50a0064be(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1bf6bad4510211d62745017b64367c7b
    def get_inputs(self):
        return [
            paddle.uniform([1, 15, 128, 128], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_be637941920053b9a665d4ae83872a4f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1bf6bad4510211d62745017b64367c7b
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 128, 128], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f67265e2f27a0574b7578de01067cf63(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1bf6bad4510211d62745017b64367c7b
    def get_inputs(self):
        return [
            paddle.uniform([1, 91, 128, 128], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_96804b7c30677abdfe3883b69eb904ea(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1bf6bad4510211d62745017b64367c7b
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 34, 34], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c90b3ae913e717db5aaef1c3edd2ad1c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1bf6bad4510211d62745017b64367c7b
    def get_inputs(self):
        return [
            paddle.uniform([1, 68, 34, 34], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c8629d59c1bac4a8c4eea713b46644ef(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1bf6bad4510211d62745017b64367c7b
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 76, 76], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6f7db0c8f9d6b20a54035541b6f0e9ec(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1bf6bad4510211d62745017b64367c7b
    def get_inputs(self):
        return [
            paddle.uniform([1, 68, 76, 76], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_643871720813fc68c1be6d874d66e380(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.flatten(input_0, 0, 1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_00bffd965ca27f8de4c8f3fd1fdd2c13(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_643871720813fc68c1be6d874d66e380
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 64, 128], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_406dfd80a1fe11fbf2de3d1e327d8022(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1bf6bad4510211d62745017b64367c7b
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 24, 24], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a7d7a9f500b6ed78d384b80c523b369a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1bf6bad4510211d62745017b64367c7b
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 24, 24], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9f56b73be1b1bece2c16a21744ba62af(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1bf6bad4510211d62745017b64367c7b
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 24, 24], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_5f472fd7eb1ae2a0503abc8d33064e2a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1bf6bad4510211d62745017b64367c7b
    def get_inputs(self):
        return [
            paddle.uniform([8, 320, 8, 32], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6e20802991417f65155555f7c2d0f633(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1bf6bad4510211d62745017b64367c7b
    def get_inputs(self):
        return [
            paddle.uniform([8, 160, 8, 32], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ae83c13e788289491cbeb3013ee4c42b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1bf6bad4510211d62745017b64367c7b
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 36, 36], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3f2634c73d3072db10fb6f7a23ed3a30(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1bf6bad4510211d62745017b64367c7b
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 36, 36], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_5ee9e6958a4c4144e18656d7cb680dd9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1bf6bad4510211d62745017b64367c7b
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 36, 36], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f5677c40391e8ea96cb739d67277910c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1bf6bad4510211d62745017b64367c7b
    def get_inputs(self):
        return [
            paddle.uniform([64, 64, 32, 8], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_58999aec7c5e1193e16814c7de21c23d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1bf6bad4510211d62745017b64367c7b
    def get_inputs(self):
        return [
            paddle.uniform([1, 15, 16, 16], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f8a52a7e27848d2d161edb5e3379008c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1bf6bad4510211d62745017b64367c7b
    def get_inputs(self):
        return [
            paddle.uniform([1, 15, 64, 64], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_5a9d6404cf0affaa2c688858d7733c51(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1bf6bad4510211d62745017b64367c7b
    def get_inputs(self):
        return [
            paddle.uniform([16, 128, 16, 32], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_76362b8d422ad609daf0ec1a2359b77c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1bf6bad4510211d62745017b64367c7b
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 13, 13], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_7576ba3f6a495463ccc985657fb074bb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1bf6bad4510211d62745017b64367c7b
    def get_inputs(self):
        return [
            paddle.uniform([1, 68, 13, 13], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b38f218950d59ab3c60763db3c290466(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4be34f96ca4d3836f254ff4cb1e59fb1
    def get_inputs(self):
        return [
            paddle.uniform([390, 64, 7, 7], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_045e1e8886cc9545ec430561e744fa52(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1bf6bad4510211d62745017b64367c7b
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 20, 20], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_7ae8a78e3976918687eb390b7f3e148b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1bf6bad4510211d62745017b64367c7b
    def get_inputs(self):
        return [
            paddle.uniform([1, 68, 20, 20], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_130403d42aeec4a49ed4197656c7d534(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1bf6bad4510211d62745017b64367c7b
    def get_inputs(self):
        return [
            paddle.uniform([8, 160, 16, 32], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_587f8d4a953166fa1f74d6b76e318891(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.flatten(input_0, 1, 2), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_fb3f9125de8fd6c8391856e24ca60ef1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_587f8d4a953166fa1f74d6b76e318891
    def get_inputs(self):
        return [
            paddle.uniform([43, 768, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_827f64a2d1500030b98e890c055d1c1a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1bf6bad4510211d62745017b64367c7b
    def get_inputs(self):
        return [
            paddle.uniform([1, 768, 32, 32], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_16dd7cc85ca628c79cf577c08f36ddad(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1bf6bad4510211d62745017b64367c7b
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 200, 304], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_16dd7cc85ca628c79cf577c08f36ddad(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1bf6bad4510211d62745017b64367c7b
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 200, 304], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ddc0158a1ce58134ca0f999028f5c3fc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1bf6bad4510211d62745017b64367c7b
    def get_inputs(self):
        return [
            paddle.uniform([1, 5, 8, 8], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9cb8bccf690f261fd3e467935b85c0cc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1bf6bad4510211d62745017b64367c7b
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 40, 40], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_8e39a9afa707488049a9f2b8930f19a5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1bf6bad4510211d62745017b64367c7b
    def get_inputs(self):
        return [
            paddle.uniform([1, 68, 40, 40], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a11593b42d1b9b25ba9c461981922fa3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1bf6bad4510211d62745017b64367c7b
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 14, 14], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_53af8a290c2fdf9ad3c75bec7dfc3749(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1bf6bad4510211d62745017b64367c7b
    def get_inputs(self):
        return [
            paddle.uniform([1, 68, 14, 14], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_7cb6c9421e94e952e13529aad61de002(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1bf6bad4510211d62745017b64367c7b
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 48, 48], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_75d6a308f9dd5bd740e50db3f34ec3cf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1bf6bad4510211d62745017b64367c7b
    def get_inputs(self):
        return [
            paddle.uniform([1, 68, 48, 48], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_47fcc4367b694c3cca9eb2c6fb1b4121(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1bf6bad4510211d62745017b64367c7b
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 21, 21], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c546e19f75600dc882dafb685715bbe6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1bf6bad4510211d62745017b64367c7b
    def get_inputs(self):
        return [
            paddle.uniform([1, 68, 21, 21], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_96804b7c30677abdfe3883b69eb904ea(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1bf6bad4510211d62745017b64367c7b
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 34, 34], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_1b857d11dfa47eb766927204a79141a3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1bf6bad4510211d62745017b64367c7b
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 34, 34], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_dd92605e9e88f777119a238c9724ae29(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1bf6bad4510211d62745017b64367c7b
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 34, 34], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_43c9d0d7c954da527b8cbdd549847a81(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1bf6bad4510211d62745017b64367c7b
    def get_inputs(self):
        return [
            paddle.uniform([1, 1280, 32, 128], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_65437b21dec7555ba36d23ead53c1f84(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1bf6bad4510211d62745017b64367c7b
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 256, 256], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9dfba762590583fd57584445e43b925f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1bf6bad4510211d62745017b64367c7b
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 18, 18], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_1e787f5e8795179eed30249ddc9d0ff9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1bf6bad4510211d62745017b64367c7b
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 18, 18], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_74e2bb1c5a1c1d49ab5b87d97e70bfd8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1bf6bad4510211d62745017b64367c7b
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 18, 18], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3b8a1adbe14e01f670629c49745dbe00(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4be34f96ca4d3836f254ff4cb1e59fb1
    def get_inputs(self):
        return [
            paddle.uniform([43, 2048, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6aeb837c9bc05fb6f246516ed2997a98(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1bf6bad4510211d62745017b64367c7b
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 17, 17], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e9fff45bbf07805339b1dc8d85c01b58(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1bf6bad4510211d62745017b64367c7b
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 17, 17], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a22e88e1260f47281167930570863d06(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1bf6bad4510211d62745017b64367c7b
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 17, 17], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_ad76feb3168f36760a514d260a055899(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.flatten(input_0, 0, 1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_f8ad85e1ff482bacf2627a4aacb5942d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ad76feb3168f36760a514d260a055899
    def get_inputs(self):
        return [
            paddle.uniform([1, 100, 80], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_5848b0a62d1692967a9e00e0f99072da(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ad76feb3168f36760a514d260a055899
    def get_inputs(self):
        return [
            paddle.uniform([1, 100, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_2243f0384ded66d542bd2758e733f7d4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1bf6bad4510211d62745017b64367c7b
    def get_inputs(self):
        return [
            paddle.uniform([6, 96, 96, 96], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3b44916f89b186740631ea1ebdd1bcbd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1bf6bad4510211d62745017b64367c7b
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 56, 56], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_37e32fd6ab6628a0a20f5f5b6eaf48c8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1bf6bad4510211d62745017b64367c7b
    def get_inputs(self):
        return [
            paddle.uniform([1, 68, 56, 56], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f8a52a7e27848d2d161edb5e3379008c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1bf6bad4510211d62745017b64367c7b
    def get_inputs(self):
        return [
            paddle.uniform([1, 15, 64, 64], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9d2039c2fe8a5d990412c8bcc251c5c5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1bf6bad4510211d62745017b64367c7b
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 64, 64], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_833ca5a7bd86e0f68d1d8e075b27100b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1bf6bad4510211d62745017b64367c7b
    def get_inputs(self):
        return [
            paddle.uniform([1, 91, 64, 64], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b392a1f1362ebbf5a3fb2d9708474cd1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1bf6bad4510211d62745017b64367c7b
    def get_inputs(self):
        return [
            paddle.uniform([1, 20, 14, 14], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_53af8a290c2fdf9ad3c75bec7dfc3749(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1bf6bad4510211d62745017b64367c7b
    def get_inputs(self):
        return [
            paddle.uniform([1, 68, 14, 14], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b43a435a07b2280f1524447772de290b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ad76feb3168f36760a514d260a055899
    def get_inputs(self):
        return [
            paddle.uniform([1, 300, 80], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_1825ac8782e68c6af004a4020050634b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ad76feb3168f36760a514d260a055899
    def get_inputs(self):
        return [
            paddle.uniform([1, 300, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6aeb837c9bc05fb6f246516ed2997a98(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1bf6bad4510211d62745017b64367c7b
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 17, 17], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6f046fc6b3777986ab6d03a2edc9a33d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1bf6bad4510211d62745017b64367c7b
    def get_inputs(self):
        return [
            paddle.uniform([1, 68, 17, 17], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_7ca661a618930261d98f5b392895966c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1bf6bad4510211d62745017b64367c7b
    def get_inputs(self):
        return [
            paddle.uniform([43, 192, 28, 28], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e352d7137846cbdfe2941ce906d4d6e9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1bf6bad4510211d62745017b64367c7b
    def get_inputs(self):
        return [
            paddle.uniform([43, 384, 14, 14], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d14ffa50a98841eea751968bf9e499e9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1bf6bad4510211d62745017b64367c7b
    def get_inputs(self):
        return [
            paddle.uniform([11, 192, 28, 28], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_5450c722abc90c57165f57cf4626b3fa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a71b4b0ddcf6f0f537f53ec17c571edc
    def get_inputs(self):
        return [
            paddle.to_tensor([[[6], [6]]], dtype='int32').reshape([1, 2, 1]),
        ]


class TestPrimitiveOp_d0121e951d333a7e7759fed2287fc702(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_eafb95a73f2f5fc792a56e4f0b174a0b
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1, 3549], dtype='int64'),
        ]


class TestPrimitiveOp_00bffd965ca27f8de4c8f3fd1fdd2c13(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_643871720813fc68c1be6d874d66e380
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 64, 128], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a6c2b7c8043404214e9dcb814b9f4d33(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4be34f96ca4d3836f254ff4cb1e59fb1
    def get_inputs(self):
        return [
            paddle.uniform([22, 1, 1, 2048], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_75e0c7e5b6b08d28ce6713f78666e7eb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1bf6bad4510211d62745017b64367c7b
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 46, 46], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_32a53b183a88a35eeedbc92b5aca160b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1bf6bad4510211d62745017b64367c7b
    def get_inputs(self):
        return [
            paddle.uniform([1, 68, 46, 46], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e18368ca924a14d069a08adacbdbb1e1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1bf6bad4510211d62745017b64367c7b
    def get_inputs(self):
        return [
            paddle.uniform([1, 15, 32, 32], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_068cad283dde9c2a4f3e277140b32fce(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1bf6bad4510211d62745017b64367c7b
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 32, 32], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ac9cc2ca95289bfcac2a209e0e0a55ee(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1bf6bad4510211d62745017b64367c7b
    def get_inputs(self):
        return [
            paddle.uniform([1, 91, 32, 32], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_00bffd965ca27f8de4c8f3fd1fdd2c13(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_643871720813fc68c1be6d874d66e380
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 64, 128], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ed12ae52d35680a93c62ba7adca2c113(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1bf6bad4510211d62745017b64367c7b
    def get_inputs(self):
        return [
            paddle.uniform([43, 96, 56, 56], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_184c52ae8d2227fefd87b6f35678a890(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1bf6bad4510211d62745017b64367c7b
    def get_inputs(self):
        return [
            paddle.uniform([10, 256, 2, 80], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e18368ca924a14d069a08adacbdbb1e1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1bf6bad4510211d62745017b64367c7b
    def get_inputs(self):
        return [
            paddle.uniform([1, 15, 32, 32], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_068cad283dde9c2a4f3e277140b32fce(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1bf6bad4510211d62745017b64367c7b
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 32, 32], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ac9cc2ca95289bfcac2a209e0e0a55ee(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1bf6bad4510211d62745017b64367c7b
    def get_inputs(self):
        return [
            paddle.uniform([1, 91, 32, 32], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_06a9687070108a986c56dcf8780141bf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1bf6bad4510211d62745017b64367c7b
    def get_inputs(self):
        return [
            paddle.uniform([11, 96, 56, 56], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_8c49f1b9adaff854ab258e310e186c2a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a71b4b0ddcf6f0f537f53ec17c571edc
    def get_inputs(self):
        return [
            paddle.to_tensor([[[3]]], dtype='int32').reshape([1, 1, 1]),
        ]


class TestPrimitiveOp_392e0202dd323c1878ac40adfdc52886(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_eafb95a73f2f5fc792a56e4f0b174a0b
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1, 4116], dtype='int64'),
        ]


class TestPrimitiveOp_4700ce74ebb0290e4775725c59fe7ed2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1bf6bad4510211d62745017b64367c7b
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 80, 80], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b1eaa71adf1b804b0acb8eb282bc19c6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1bf6bad4510211d62745017b64367c7b
    def get_inputs(self):
        return [
            paddle.uniform([1, 68, 80, 80], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_38ddd76ba666e02c11dfac0e4fb459b0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1bf6bad4510211d62745017b64367c7b
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 60, 60], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_91b7310b7d602f315648d1ba9608d170(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1bf6bad4510211d62745017b64367c7b
    def get_inputs(self):
        return [
            paddle.uniform([1, 68, 60, 60], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ccf7c7d92c59f461678840e021914869(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1bf6bad4510211d62745017b64367c7b
    def get_inputs(self):
        return [
            paddle.uniform([16, 64, 16, 32], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a1968726c8de302233c6fbb0d935d8df(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1bf6bad4510211d62745017b64367c7b
    def get_inputs(self):
        return [
            paddle.uniform([10, 128, 4, 25], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d84b1ccce0049bb52419f95e562acf62(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1bf6bad4510211d62745017b64367c7b
    def get_inputs(self):
        return [
            paddle.uniform([1, 20, 56, 56], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_37e32fd6ab6628a0a20f5f5b6eaf48c8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1bf6bad4510211d62745017b64367c7b
    def get_inputs(self):
        return [
            paddle.uniform([1, 68, 56, 56], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_64db5737becf046b9cb1cf2bf3738155(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4be34f96ca4d3836f254ff4cb1e59fb1
    def get_inputs(self):
        return [
            paddle.uniform([22, 2048, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_343abc89eb21154a2262af8ad243f272(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1bf6bad4510211d62745017b64367c7b
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 96, 96], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_aab59e7b3a0acd45db541c5d38670dfa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1bf6bad4510211d62745017b64367c7b
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 96, 96], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d1fd7a007ef629574adf4f631f83dd57(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1bf6bad4510211d62745017b64367c7b
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 96, 96], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_edf2d260a393a90d18ab5841f7befc72(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1bf6bad4510211d62745017b64367c7b
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 52, 52], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9a64f7e7f6776f223eef20d42b589b62(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1bf6bad4510211d62745017b64367c7b
    def get_inputs(self):
        return [
            paddle.uniform([1, 76, 52, 52], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_29ca4fed1ef3839e7778bbed98c9e116(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4be34f96ca4d3836f254ff4cb1e59fb1
    def get_inputs(self):
        return [
            paddle.uniform([22, 1000, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_5001c2412a25a14d3f11963db666f573(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1bf6bad4510211d62745017b64367c7b
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 128, 256], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_7fd0b901d5932e520646b5cd61ce5f89(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1bf6bad4510211d62745017b64367c7b
    def get_inputs(self):
        return [
            paddle.uniform([10, 128, 4, 80], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e18368ca924a14d069a08adacbdbb1e1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1bf6bad4510211d62745017b64367c7b
    def get_inputs(self):
        return [
            paddle.uniform([1, 15, 32, 32], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4700ce74ebb0290e4775725c59fe7ed2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1bf6bad4510211d62745017b64367c7b
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 80, 80], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_403a7759274f5910124d82cbf24fdebc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1bf6bad4510211d62745017b64367c7b
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 80, 80], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a437ebccc2303b13baaedc4a9cae086a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1bf6bad4510211d62745017b64367c7b
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 80, 80], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_76362b8d422ad609daf0ec1a2359b77c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1bf6bad4510211d62745017b64367c7b
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 13, 13], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_7576ba3f6a495463ccc985657fb074bb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1bf6bad4510211d62745017b64367c7b
    def get_inputs(self):
        return [
            paddle.uniform([1, 68, 13, 13], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3d8e7d0391d526bcf11d0705eb93d012(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1bf6bad4510211d62745017b64367c7b
    def get_inputs(self):
        return [
            paddle.uniform([11, 768, 7, 7], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ae1e4c190fa4b2bb0525f8bf8fec007f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1bf6bad4510211d62745017b64367c7b
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 26, 26], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ff2e04a15fb3c1e3d85d7fe3c2bafa8b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1bf6bad4510211d62745017b64367c7b
    def get_inputs(self):
        return [
            paddle.uniform([1, 68, 26, 26], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_eff9fa5da4b4090da25cf151fc0a21db(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1bf6bad4510211d62745017b64367c7b
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 23, 23], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_75c059779dc49b4ca30c2034ea0affe6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1bf6bad4510211d62745017b64367c7b
    def get_inputs(self):
        return [
            paddle.uniform([1, 68, 23, 23], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f8a52a7e27848d2d161edb5e3379008c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1bf6bad4510211d62745017b64367c7b
    def get_inputs(self):
        return [
            paddle.uniform([1, 15, 64, 64], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9d2039c2fe8a5d990412c8bcc251c5c5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1bf6bad4510211d62745017b64367c7b
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 64, 64], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_833ca5a7bd86e0f68d1d8e075b27100b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1bf6bad4510211d62745017b64367c7b
    def get_inputs(self):
        return [
            paddle.uniform([1, 91, 64, 64], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a93daeca6628e67a5242c94060e96e70(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1bf6bad4510211d62745017b64367c7b
    def get_inputs(self):
        return [
            paddle.uniform([8, 256, 8, 16], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_55b8aaa0e0453ee8b26e21e50a0064be(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1bf6bad4510211d62745017b64367c7b
    def get_inputs(self):
        return [
            paddle.uniform([1, 15, 128, 128], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_be637941920053b9a665d4ae83872a4f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1bf6bad4510211d62745017b64367c7b
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 128, 128], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f67265e2f27a0574b7578de01067cf63(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1bf6bad4510211d62745017b64367c7b
    def get_inputs(self):
        return [
            paddle.uniform([1, 91, 128, 128], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_55b8aaa0e0453ee8b26e21e50a0064be(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1bf6bad4510211d62745017b64367c7b
    def get_inputs(self):
        return [
            paddle.uniform([1, 15, 128, 128], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_be637941920053b9a665d4ae83872a4f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1bf6bad4510211d62745017b64367c7b
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 128, 128], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f67265e2f27a0574b7578de01067cf63(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1bf6bad4510211d62745017b64367c7b
    def get_inputs(self):
        return [
            paddle.uniform([1, 91, 128, 128], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_34defd6b7a0ef545167b963c2337b4fd(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.flatten(input_0, 0, 1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_53322641dd81a87ca554fe05a5917b43(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_34defd6b7a0ef545167b963c2337b4fd
    def get_inputs(self):
        return [
            paddle.uniform([24, 24], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_53322641dd81a87ca554fe05a5917b43(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_34defd6b7a0ef545167b963c2337b4fd
    def get_inputs(self):
        return [
            paddle.uniform([24, 24], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9cb8bccf690f261fd3e467935b85c0cc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1bf6bad4510211d62745017b64367c7b
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 40, 40], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_294f201daa622e91bd598f5383066742(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1bf6bad4510211d62745017b64367c7b
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 40, 40], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_1c064fa6adb5c7d9889866dcf273bce7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1bf6bad4510211d62745017b64367c7b
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 40, 40], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a68f40cb6634133da8d1fcec47996248(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1bf6bad4510211d62745017b64367c7b
    def get_inputs(self):
        return [
            paddle.uniform([11, 384, 14, 14], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_340a369b1152626551df8e15cb5fb933(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1bf6bad4510211d62745017b64367c7b
    def get_inputs(self):
        return [
            paddle.uniform([1, 5, 64, 64], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_7b6af94be5d5bcdd1a395b6062462813(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1bf6bad4510211d62745017b64367c7b
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 19, 19], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3e641dbd567f4f455db15ed84a85b1dc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1bf6bad4510211d62745017b64367c7b
    def get_inputs(self):
        return [
            paddle.uniform([1, 68, 19, 19], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e18368ca924a14d069a08adacbdbb1e1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1bf6bad4510211d62745017b64367c7b
    def get_inputs(self):
        return [
            paddle.uniform([1, 15, 32, 32], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_068cad283dde9c2a4f3e277140b32fce(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1bf6bad4510211d62745017b64367c7b
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 32, 32], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ac9cc2ca95289bfcac2a209e0e0a55ee(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1bf6bad4510211d62745017b64367c7b
    def get_inputs(self):
        return [
            paddle.uniform([1, 91, 32, 32], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_841c77a4c5954fad7ed3c04546cf209b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1bf6bad4510211d62745017b64367c7b
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 32, 32], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_068cad283dde9c2a4f3e277140b32fce(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1bf6bad4510211d62745017b64367c7b
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 32, 32], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_08052b4b9b8bec2fffa0fc414e5e5dac(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1bf6bad4510211d62745017b64367c7b
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 32, 32], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_5848b0a62d1692967a9e00e0f99072da(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ad76feb3168f36760a514d260a055899
    def get_inputs(self):
        return [
            paddle.uniform([1, 100, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_634dc605580f182db5625265cf24830a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_587f8d4a953166fa1f74d6b76e318891
    def get_inputs(self):
        return [
            paddle.uniform([11, 768, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a11695956b67cbd9748e74de86aac70d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1bf6bad4510211d62745017b64367c7b
    def get_inputs(self):
        return [
            paddle.uniform([10, 256, 2, 25], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b453ea4d4569e359c78186ddeda53cce(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1bf6bad4510211d62745017b64367c7b
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 136, 160], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b453ea4d4569e359c78186ddeda53cce(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1bf6bad4510211d62745017b64367c7b
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 136, 160], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_00bffd965ca27f8de4c8f3fd1fdd2c13(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_643871720813fc68c1be6d874d66e380
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 64, 128], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_1825ac8782e68c6af004a4020050634b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ad76feb3168f36760a514d260a055899
    def get_inputs(self):
        return [
            paddle.uniform([1, 300, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ae1e4c190fa4b2bb0525f8bf8fec007f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1bf6bad4510211d62745017b64367c7b
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 26, 26], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ff2e04a15fb3c1e3d85d7fe3c2bafa8b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1bf6bad4510211d62745017b64367c7b
    def get_inputs(self):
        return [
            paddle.uniform([1, 68, 26, 26], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9aec25aabce55faa679f761b974e4982(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1bf6bad4510211d62745017b64367c7b
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 68, 68], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_85d9fa38e6c8e9a829e246fed814b5fc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1bf6bad4510211d62745017b64367c7b
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 68, 68], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3a6838d731dba4156462bc0108bc114a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1bf6bad4510211d62745017b64367c7b
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 68, 68], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3695037e95d0f54390e7ea1239a0b72d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4be34f96ca4d3836f254ff4cb1e59fb1
    def get_inputs(self):
        return [
            paddle.uniform([11, 704, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b8b42864fe81837aaca6a7c0e1875019(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1bf6bad4510211d62745017b64367c7b
    def get_inputs(self):
        return [
            paddle.uniform([1, 20, 40, 40], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_8e39a9afa707488049a9f2b8930f19a5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1bf6bad4510211d62745017b64367c7b
    def get_inputs(self):
        return [
            paddle.uniform([1, 68, 40, 40], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_7f6632b532adc3dce0f38220b70d2338(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1bf6bad4510211d62745017b64367c7b
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 72, 72], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c2d96b369e221195b78e5cde5da6699c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1bf6bad4510211d62745017b64367c7b
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 72, 72], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_bcd6f1d16d97437935e2b110c02d7540(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1bf6bad4510211d62745017b64367c7b
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 72, 72], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_357f2eee9b6e4c68f6a45f928dee2b64(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1bf6bad4510211d62745017b64367c7b
    def get_inputs(self):
        return [
            paddle.uniform([1, 5, 128, 128], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_fdad369f5cd3e97bf5061e5fd69e8624(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1bf6bad4510211d62745017b64367c7b
    def get_inputs(self):
        return [
            paddle.uniform([1, 20, 10, 10], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_46777152a86ca6b661ad507329d5ff00(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1bf6bad4510211d62745017b64367c7b
    def get_inputs(self):
        return [
            paddle.uniform([1, 68, 10, 10], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3d8e7d0391d526bcf11d0705eb93d012(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1bf6bad4510211d62745017b64367c7b
    def get_inputs(self):
        return [
            paddle.uniform([11, 768, 7, 7], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6cc3d734859399267944a182df628291(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1bf6bad4510211d62745017b64367c7b
    def get_inputs(self):
        return [
            paddle.uniform([4, 96, 96, 96], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_566b724f5d8b99afd5a21306bf5dd319(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1bf6bad4510211d62745017b64367c7b
    def get_inputs(self):
        return [
            paddle.uniform([1, 20, 20, 20], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_7ae8a78e3976918687eb390b7f3e148b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1bf6bad4510211d62745017b64367c7b
    def get_inputs(self):
        return [
            paddle.uniform([1, 68, 20, 20], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_76e79bb6d022c3cc091f5383052b855c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1bf6bad4510211d62745017b64367c7b
    def get_inputs(self):
        return [
            paddle.uniform([43, 768, 7, 7], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_406dfd80a1fe11fbf2de3d1e327d8022(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1bf6bad4510211d62745017b64367c7b
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 24, 24], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_8f4cb3efe58db72935adf1f5ba125bdc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1bf6bad4510211d62745017b64367c7b
    def get_inputs(self):
        return [
            paddle.uniform([1, 68, 24, 24], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_862a3479eecff28506a88d56f91e719b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1bf6bad4510211d62745017b64367c7b
    def get_inputs(self):
        return [
            paddle.uniform([1, 5, 16, 16], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e352d7137846cbdfe2941ce906d4d6e9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1bf6bad4510211d62745017b64367c7b
    def get_inputs(self):
        return [
            paddle.uniform([43, 384, 14, 14], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b0ed731929c8f1b10e1bede7d3ab249d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1bf6bad4510211d62745017b64367c7b
    def get_inputs(self):
        return [
            paddle.uniform([1, 1280, 32, 64], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_76e79bb6d022c3cc091f5383052b855c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1bf6bad4510211d62745017b64367c7b
    def get_inputs(self):
        return [
            paddle.uniform([43, 768, 7, 7], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_edf2d260a393a90d18ab5841f7befc72(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1bf6bad4510211d62745017b64367c7b
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 52, 52], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_bff9935367f55f43614733531941911b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1bf6bad4510211d62745017b64367c7b
    def get_inputs(self):
        return [
            paddle.uniform([1, 68, 52, 52], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_55b8aaa0e0453ee8b26e21e50a0064be(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1bf6bad4510211d62745017b64367c7b
    def get_inputs(self):
        return [
            paddle.uniform([1, 15, 128, 128], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_045e1e8886cc9545ec430561e744fa52(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1bf6bad4510211d62745017b64367c7b
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 20, 20], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d83e5422387a17a4b47e1a13ec0bba4a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1bf6bad4510211d62745017b64367c7b
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 20, 20], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4fb886c2b57e4df86754e42e24c8af4c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1bf6bad4510211d62745017b64367c7b
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 20, 20], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c2829f076e0cfe7f313d67459edb1ba0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1bf6bad4510211d62745017b64367c7b
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 92, 92], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_47c45bfcdef4c0f1734f9340f7d9d16c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1bf6bad4510211d62745017b64367c7b
    def get_inputs(self):
        return [
            paddle.uniform([1, 68, 92, 92], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_55b8aaa0e0453ee8b26e21e50a0064be(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1bf6bad4510211d62745017b64367c7b
    def get_inputs(self):
        return [
            paddle.uniform([1, 15, 128, 128], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_be637941920053b9a665d4ae83872a4f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1bf6bad4510211d62745017b64367c7b
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 128, 128], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f67265e2f27a0574b7578de01067cf63(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1bf6bad4510211d62745017b64367c7b
    def get_inputs(self):
        return [
            paddle.uniform([1, 91, 128, 128], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f8a52a7e27848d2d161edb5e3379008c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1bf6bad4510211d62745017b64367c7b
    def get_inputs(self):
        return [
            paddle.uniform([1, 15, 64, 64], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9d2039c2fe8a5d990412c8bcc251c5c5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1bf6bad4510211d62745017b64367c7b
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 64, 64], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_833ca5a7bd86e0f68d1d8e075b27100b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1bf6bad4510211d62745017b64367c7b
    def get_inputs(self):
        return [
            paddle.uniform([1, 91, 64, 64], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_38de7486d84967f4b7bbb382a8f59d4e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1bf6bad4510211d62745017b64367c7b
    def get_inputs(self):
        return [
            paddle.uniform([1, 20, 28, 28], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_33a0692871fe722c31a4ee446f4dbfe0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1bf6bad4510211d62745017b64367c7b
    def get_inputs(self):
        return [
            paddle.uniform([1, 68, 28, 28], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_7e17a5a0949bb2dd968285e869173736(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1bf6bad4510211d62745017b64367c7b
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 28, 28], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_33a0692871fe722c31a4ee446f4dbfe0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1bf6bad4510211d62745017b64367c7b
    def get_inputs(self):
        return [
            paddle.uniform([1, 68, 28, 28], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a68f40cb6634133da8d1fcec47996248(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1bf6bad4510211d62745017b64367c7b
    def get_inputs(self):
        return [
            paddle.uniform([11, 384, 14, 14], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_7d37b55d5affeaad8a034718b5a5f414(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1bf6bad4510211d62745017b64367c7b
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 38, 38], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b7091a48ab344a93fa82bc3af8e73fde(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1bf6bad4510211d62745017b64367c7b
    def get_inputs(self):
        return [
            paddle.uniform([1, 68, 38, 38], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_7ca661a618930261d98f5b392895966c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1bf6bad4510211d62745017b64367c7b
    def get_inputs(self):
        return [
            paddle.uniform([43, 192, 28, 28], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e31c7bfc24e03ec95a886b3df241aa1a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1bf6bad4510211d62745017b64367c7b
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 42, 42], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f52885b8990b39097abe522fbfc3ef5e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1bf6bad4510211d62745017b64367c7b
    def get_inputs(self):
        return [
            paddle.uniform([1, 68, 42, 42], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d14ffa50a98841eea751968bf9e499e9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1bf6bad4510211d62745017b64367c7b
    def get_inputs(self):
        return [
            paddle.uniform([11, 192, 28, 28], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_45215d72f76bf1306e28782f3223f83a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1bf6bad4510211d62745017b64367c7b
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 12, 12], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_eace009bb12f5e7dbb9c42de695dcd37(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1bf6bad4510211d62745017b64367c7b
    def get_inputs(self):
        return [
            paddle.uniform([1, 68, 12, 12], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b97362c18a75128305f0ca6c67f157de(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1bf6bad4510211d62745017b64367c7b
    def get_inputs(self):
        return [
            paddle.uniform([4, 512, 8, 32], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_cc38c72c3f7f5ded34ee852b58206960(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1bf6bad4510211d62745017b64367c7b
    def get_inputs(self):
        return [
            paddle.uniform([4, 512, 4, 32], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c5928b72c80483f22551e8315cd1ed00(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1bf6bad4510211d62745017b64367c7b
    def get_inputs(self):
        return [
            paddle.uniform([1, 5, 32, 32], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_224a8139962d5f249f813431ee74c7a8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1bf6bad4510211d62745017b64367c7b
    def get_inputs(self):
        return [
            paddle.uniform([4, 256, 8, 16], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9aec25aabce55faa679f761b974e4982(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1bf6bad4510211d62745017b64367c7b
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 68, 68], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_00761ad37c67ed5f3ac6a26a019a01b4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1bf6bad4510211d62745017b64367c7b
    def get_inputs(self):
        return [
            paddle.uniform([1, 68, 68, 68], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_7cb6c9421e94e952e13529aad61de002(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1bf6bad4510211d62745017b64367c7b
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 48, 48], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_54e9e13864dd50bca3abbd1670bed092(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1bf6bad4510211d62745017b64367c7b
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 48, 48], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f99966bcd47dcbc47057e587ae5e8c38(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1bf6bad4510211d62745017b64367c7b
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 48, 48], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_470fc054d80cc4bd83b89bd6762c34a7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4be34f96ca4d3836f254ff4cb1e59fb1
    def get_inputs(self):
        return [
            paddle.uniform([43, 704, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_06a9687070108a986c56dcf8780141bf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1bf6bad4510211d62745017b64367c7b
    def get_inputs(self):
        return [
            paddle.uniform([11, 96, 56, 56], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ed12ae52d35680a93c62ba7adca2c113(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1bf6bad4510211d62745017b64367c7b
    def get_inputs(self):
        return [
            paddle.uniform([43, 96, 56, 56], dtype='float32', min=0, max=0.5),
        ]




if __name__ == '__main__':
    unittest.main()