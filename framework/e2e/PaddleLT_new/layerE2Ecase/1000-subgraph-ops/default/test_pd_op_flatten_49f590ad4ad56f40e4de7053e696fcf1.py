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


class PrimitiveOp_7e725265c28a5e7a7e4f95d9a2ef1499(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.flatten(input_0, 2, 3), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 91, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_0e40a5c1be3b31a99f08239902efb014(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7e725265c28a5e7a7e4f95d9a2ef1499
    def get_inputs(self):
        return [
            paddle.uniform([1, 91, 32, 32], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_0cd021692c58fe9cf7d9545599a47c65(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.flatten(input_0, 2, 3), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 128, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_6a9a6895f3a6d2c2607d1a7594ac71f7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0cd021692c58fe9cf7d9545599a47c65
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


class PrimitiveOp_97c9dd9d0add9205a9f0d961ed6a22ee(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.flatten(input_0, 2, 3), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 68, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_8ec2a5c281075cf144bbf878634bc6bf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_97c9dd9d0add9205a9f0d961ed6a22ee
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


class TestPrimitiveOp_4ac84e98943dad1063f5b5503d9f1283(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_97c9dd9d0add9205a9f0d961ed6a22ee
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


class TestPrimitiveOp_58130281d2efc5f0cb55c31d9e125e87(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_97c9dd9d0add9205a9f0d961ed6a22ee
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


class TestPrimitiveOp_2abab84bc129615ce31fd9eaa5600ec2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_97c9dd9d0add9205a9f0d961ed6a22ee
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


class PrimitiveOp_d9751b0b0f260fdd593da6c4b5ae3010(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.flatten(input_0, 2, 3), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 320, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_e4d1d822c993bbb1eb27429f790cafc3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d9751b0b0f260fdd593da6c4b5ae3010
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


class TestPrimitiveOp_e7963c6cd558e4d76895cd669285f6df(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7e725265c28a5e7a7e4f95d9a2ef1499
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


class PrimitiveOp_97610cebc76fc5fe815b9b2505107f26(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.flatten(input_0, 2, 3), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 76, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_2f018ab85f9e581ec1d5cc8f4c39a1d2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_97610cebc76fc5fe815b9b2505107f26
    def get_inputs(self):
        return [
            paddle.uniform([1, 76, 26, 26], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_73624ddd9f8dae7c3d4f60bfe62eb639(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.flatten(input_0, 1, 3), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_cda8b6664f9a339f1bedc856d6540726(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_73624ddd9f8dae7c3d4f60bfe62eb639
    def get_inputs(self):
        return [
            paddle.uniform([11, 2048, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_10481b2bab1d35fe0e3cc18f7681f4f4(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.flatten(input_0, 1, 3), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 1000, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_3ead67afa4bba0e2e80287688ba050d6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_10481b2bab1d35fe0e3cc18f7681f4f4
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


class TestPrimitiveOp_d1a6b6a62580ff7dd557bbbcd0581b14(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_97c9dd9d0add9205a9f0d961ed6a22ee
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


class TestPrimitiveOp_f456d7b29f047c91d34e42e29ee54e42(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_97c9dd9d0add9205a9f0d961ed6a22ee
    def get_inputs(self):
        return [
            paddle.uniform([1, 68, 52, 52], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_bef188218c441198e62140fdb9816a3c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.flatten(input_0, 2, 3), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 15, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_08e9b89e8faa124fd2661e1b82b3714e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bef188218c441198e62140fdb9816a3c
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


class TestPrimitiveOp_b0b82675a34920347a8038ab2fa6b959(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_97610cebc76fc5fe815b9b2505107f26
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


class TestPrimitiveOp_4d047ae5807f8da681bfdf80b93c51b8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7e725265c28a5e7a7e4f95d9a2ef1499
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


class TestPrimitiveOp_48c88dd6c11a415846944e4f0a74e28a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_97c9dd9d0add9205a9f0d961ed6a22ee
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


class TestPrimitiveOp_ad8a8129c2fa55233c04f83cba59e987(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_97c9dd9d0add9205a9f0d961ed6a22ee
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


class TestPrimitiveOp_017290cbeea95c9de380f96daff66a57(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d9751b0b0f260fdd593da6c4b5ae3010
    def get_inputs(self):
        return [
            paddle.uniform([8, 320, 8, 32], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_0dd3653f862918d17f14fc8183ff9910(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.flatten(input_0, 2, 3), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 160, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_17fbb48d06ba07e3e970004a4ae85aa5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0dd3653f862918d17f14fc8183ff9910
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


class PrimitiveOp_86730eeea6d75dcc22e62eaa995c1a5f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.flatten(input_0, 2, 3), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 64, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_d8a8f2a4172d02293f2448e83070c5c4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_86730eeea6d75dcc22e62eaa995c1a5f
    def get_inputs(self):
        return [
            paddle.uniform([64, 64, 32, 8], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_43e8fbec6d7d38b34f5b966eb0f13fef(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bef188218c441198e62140fdb9816a3c
    def get_inputs(self):
        return [
            paddle.uniform([1, 15, 16, 16], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_adac0b22d1725d764a83e809e1b7af69(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bef188218c441198e62140fdb9816a3c
    def get_inputs(self):
        return [
            paddle.uniform([1, 15, 64, 64], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_15eb86ad8b7df52cdaaf44ac8ce3e623(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0cd021692c58fe9cf7d9545599a47c65
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


class TestPrimitiveOp_49dcac2ce25258505f31a7acac6f9e45(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_97c9dd9d0add9205a9f0d961ed6a22ee
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


class TestPrimitiveOp_2f19b56fb2361aa253d6cca4e33def8f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_97c9dd9d0add9205a9f0d961ed6a22ee
    def get_inputs(self):
        return [
            paddle.uniform([1, 68, 20, 20], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4f36d13dfcf95f0e3117d60fedc6a126(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0dd3653f862918d17f14fc8183ff9910
    def get_inputs(self):
        return [
            paddle.uniform([8, 160, 16, 32], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_e1afaff216534e0eca31318f89dd128e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.flatten(input_0, 1, 2), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 768, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_680db63135f64f82e31d722c5fa3db4b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e1afaff216534e0eca31318f89dd128e
    def get_inputs(self):
        return [
            paddle.uniform([43, 768, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_7518a11010ee8b5ed8e90f35f7622dc7(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.flatten(input_0, 2, 3), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 768, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_3ce240181446c9c058f7b4b9e82b0c77(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7518a11010ee8b5ed8e90f35f7622dc7
    def get_inputs(self):
        return [
            paddle.uniform([1, 768, 32, 32], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_9f24a611edb72752eae6630f313d6010(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.flatten(input_0, 2, 3), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 96, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_74e3f4c1246f0e198c181e02b0430b12(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9f24a611edb72752eae6630f313d6010
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 200, 304], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_19ff49497fdf7541ea90648cdd5aeca4(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.flatten(input_0, 2, 3), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 96, 200, 304], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_450f4fad30a2c02366f712e4abf206bd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_19ff49497fdf7541ea90648cdd5aeca4
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 200, 304], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_6d494b25d0d6cb134b2488a98f9ab641(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.flatten(input_0, 2, 3), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 5, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_7b604c951bc0a30cd5f24676454e0802(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6d494b25d0d6cb134b2488a98f9ab641
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


class TestPrimitiveOp_6ab3ec6c37a40d5a1e58309b2f9790cd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_97c9dd9d0add9205a9f0d961ed6a22ee
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


class TestPrimitiveOp_4961d99f0598170747f323352d323aa9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_97c9dd9d0add9205a9f0d961ed6a22ee
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


class TestPrimitiveOp_58130281d2efc5f0cb55c31d9e125e87(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_97c9dd9d0add9205a9f0d961ed6a22ee
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


class TestPrimitiveOp_be4cd1c6f9fec04363d809c7cdd4eef8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_97c9dd9d0add9205a9f0d961ed6a22ee
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


class PrimitiveOp_4a4a5d372a26e9b83fdaf9efbc66706f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.flatten(input_0, 2, 3), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 1280, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_881ce7241e83b0f646a0523a7ce45492(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4a4a5d372a26e9b83fdaf9efbc66706f
    def get_inputs(self):
        return [
            paddle.uniform([1, 1280, 32, 128], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_16bb6a939610b3427e3ce28818fbcb9e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.flatten(input_0, 2, 3), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 32, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_9ec709fbe387f89cb8e66607d7074ee5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_16bb6a939610b3427e3ce28818fbcb9e
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


class TestPrimitiveOp_3f6db36df0ef86794a9ba71b8283e8e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_73624ddd9f8dae7c3d4f60bfe62eb639
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


class TestPrimitiveOp_66663ee487b2b3d64e4d6ef585a8cd55(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9f24a611edb72752eae6630f313d6010
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


class TestPrimitiveOp_1bd58b6d4762c3335770ecfa67c4eb98(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_97c9dd9d0add9205a9f0d961ed6a22ee
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


class TestPrimitiveOp_e7963c6cd558e4d76895cd669285f6df(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7e725265c28a5e7a7e4f95d9a2ef1499
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


class TestPrimitiveOp_4961d99f0598170747f323352d323aa9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_97c9dd9d0add9205a9f0d961ed6a22ee
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


class TestPrimitiveOp_fa7af9c4ed60f85aa5cb2791414bb93e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_97c9dd9d0add9205a9f0d961ed6a22ee
    def get_inputs(self):
        return [
            paddle.uniform([1, 68, 17, 17], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_903e0b046a72c2e939a35351b0d4cd17(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.flatten(input_0, 2, 3), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[43, 192, 28, 28], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_baa9a63e685b49b22f2a9f745bb29860(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_903e0b046a72c2e939a35351b0d4cd17
    def get_inputs(self):
        return [
            paddle.uniform([43, 192, 28, 28], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_d3f6a8be632cf3d25098d38246f4a166(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.flatten(input_0, 2, 3), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[43, 384, 14, 14], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_8fc046f10d4dfe62a3e556e0257de6ea(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d3f6a8be632cf3d25098d38246f4a166
    def get_inputs(self):
        return [
            paddle.uniform([43, 384, 14, 14], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_2338c6ff1ed7583a883c5e3ac4a5d5a0(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.flatten(input_0, 2, 3), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 192, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_f31e844315d97e609616c48847d601ea(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2338c6ff1ed7583a883c5e3ac4a5d5a0
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


class PrimitiveOp_2fe0627fc63168eb14b78af3c5214ab6(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.flatten(input_0, 1, 3), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 1, 1, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_59c4a60efeb6edde5489a080aef0dc39(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2fe0627fc63168eb14b78af3c5214ab6
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


class TestPrimitiveOp_fcc0e9a0c2af7a1f29aaef45f09ec12d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_97c9dd9d0add9205a9f0d961ed6a22ee
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


class TestPrimitiveOp_0e40a5c1be3b31a99f08239902efb014(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7e725265c28a5e7a7e4f95d9a2ef1499
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


class TestPrimitiveOp_95a2bab4ccf2c290d94ccf9bf6c603b8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9f24a611edb72752eae6630f313d6010
    def get_inputs(self):
        return [
            paddle.uniform([43, 96, 56, 56], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_6319d7b39586c78730ad220defe84deb(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.flatten(input_0, 2, 3), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 256, 2, 80], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_4c29a719e6f7e91c4d7f120c4a3c8f67(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6319d7b39586c78730ad220defe84deb
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


class TestPrimitiveOp_0e40a5c1be3b31a99f08239902efb014(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7e725265c28a5e7a7e4f95d9a2ef1499
    def get_inputs(self):
        return [
            paddle.uniform([1, 91, 32, 32], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_bc4d7491923890a8c9ac183be1b188e9(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.flatten(input_0, 2, 3), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[11, 96, 56, 56], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_f0fef1d27e3f8f78d6b3f94fb26197d4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bc4d7491923890a8c9ac183be1b188e9
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


class TestPrimitiveOp_29b33f0d0c08aac7a0cd8e0bbc953cfd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_97c9dd9d0add9205a9f0d961ed6a22ee
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


class TestPrimitiveOp_ce6b17edb64003168b2a0a7f57985442(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_97c9dd9d0add9205a9f0d961ed6a22ee
    def get_inputs(self):
        return [
            paddle.uniform([1, 68, 60, 60], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_7b797aa76a42eca68a7f7f87a34e49f8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_86730eeea6d75dcc22e62eaa995c1a5f
    def get_inputs(self):
        return [
            paddle.uniform([16, 64, 16, 32], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_b9697182b139531b6ee6904fd29e69fc(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.flatten(input_0, 2, 3), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 128, 4, 25], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_34ee942d538141d101ab16b9127314f7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b9697182b139531b6ee6904fd29e69fc
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


class TestPrimitiveOp_1bd58b6d4762c3335770ecfa67c4eb98(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_97c9dd9d0add9205a9f0d961ed6a22ee
    def get_inputs(self):
        return [
            paddle.uniform([1, 68, 56, 56], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a5656d1a6e473aed4125041eeed34440(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_73624ddd9f8dae7c3d4f60bfe62eb639
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


class TestPrimitiveOp_6f91b58a3f1cb5a092ac483b7e4efcf1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_97610cebc76fc5fe815b9b2505107f26
    def get_inputs(self):
        return [
            paddle.uniform([1, 76, 52, 52], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_59c3aaa9f7541470d1e8eed90340bdb7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_10481b2bab1d35fe0e3cc18f7681f4f4
    def get_inputs(self):
        return [
            paddle.uniform([22, 1000, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_398764aae6a887550eacffec524cd38b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_16bb6a939610b3427e3ce28818fbcb9e
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 128, 256], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_79c71b8f13db7b0d84b8f596d92420c7(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.flatten(input_0, 2, 3), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 128, 4, 80], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_012eae47089ec8636e2cb4586bbaacbe(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_79c71b8f13db7b0d84b8f596d92420c7
    def get_inputs(self):
        return [
            paddle.uniform([10, 128, 4, 80], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_77a9bf9bf72fa01f9340cde8c33ffe2a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bef188218c441198e62140fdb9816a3c
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


class TestPrimitiveOp_49dcac2ce25258505f31a7acac6f9e45(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_97c9dd9d0add9205a9f0d961ed6a22ee
    def get_inputs(self):
        return [
            paddle.uniform([1, 68, 13, 13], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_59eaa7bc925989f5e159019ccd4b74da(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7518a11010ee8b5ed8e90f35f7622dc7
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


class TestPrimitiveOp_2c83db2f7440036ec58b2353678efcb5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_97c9dd9d0add9205a9f0d961ed6a22ee
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


class TestPrimitiveOp_99539ff3b90fa7145ed68b8f80ec01c2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_97c9dd9d0add9205a9f0d961ed6a22ee
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


class TestPrimitiveOp_e7963c6cd558e4d76895cd669285f6df(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7e725265c28a5e7a7e4f95d9a2ef1499
    def get_inputs(self):
        return [
            paddle.uniform([1, 91, 64, 64], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_eb000f903e80fc402e871cadc1b93699(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.flatten(input_0, 2, 3), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 256, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_2c329c9aa3afbe1c702e1f93496a3fa0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_eb000f903e80fc402e871cadc1b93699
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


class TestPrimitiveOp_4d047ae5807f8da681bfdf80b93c51b8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7e725265c28a5e7a7e4f95d9a2ef1499
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


class TestPrimitiveOp_4d047ae5807f8da681bfdf80b93c51b8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7e725265c28a5e7a7e4f95d9a2ef1499
    def get_inputs(self):
        return [
            paddle.uniform([1, 91, 128, 128], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_099d382f010bf13e17d116d2a3f45625(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.flatten(input_0, 0, 1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[24, 24], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_b1b2b3dd63a05931009ed43377ec19a1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_099d382f010bf13e17d116d2a3f45625
    def get_inputs(self):
        return [
            paddle.uniform([24, 24], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b1b2b3dd63a05931009ed43377ec19a1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_099d382f010bf13e17d116d2a3f45625
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


class PrimitiveOp_179dfad6e6c820702206938092019741(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.flatten(input_0, 2, 3), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[11, 384, 14, 14], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_a973ea6e773bc115897f5de9bef40597(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_179dfad6e6c820702206938092019741
    def get_inputs(self):
        return [
            paddle.uniform([11, 384, 14, 14], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_364bb6f191ce485733f7d75a84af3b32(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6d494b25d0d6cb134b2488a98f9ab641
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


class TestPrimitiveOp_96d4d8bf4ffcc2fd3673e593686a2ff0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_97c9dd9d0add9205a9f0d961ed6a22ee
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


class TestPrimitiveOp_0e40a5c1be3b31a99f08239902efb014(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7e725265c28a5e7a7e4f95d9a2ef1499
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


class PrimitiveOp_a932bca24fd27301a2416490fcdde767(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.flatten(input_0, 0, 1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 100, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_9fe30a496d563580dd5823cc994567c5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a932bca24fd27301a2416490fcdde767
    def get_inputs(self):
        return [
            paddle.uniform([1, 100, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6e3874985d969c1fbb58e5cb4b43ae15(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e1afaff216534e0eca31318f89dd128e
    def get_inputs(self):
        return [
            paddle.uniform([11, 768, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_536f6d43f87721518e886d55df3e7de8(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.flatten(input_0, 2, 3), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 256, 2, 25], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_d72392c16fb1abcb537bdca7d07cb495(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_536f6d43f87721518e886d55df3e7de8
    def get_inputs(self):
        return [
            paddle.uniform([10, 256, 2, 25], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_946152afefd56ed9709c3369f36f689b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9f24a611edb72752eae6630f313d6010
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 136, 160], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_930a0d66834515bf9d3cd0678c50292a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.flatten(input_0, 2, 3), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 96, 136, 160], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_670888c6f258efcb7b27b98c1c4ae72e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_930a0d66834515bf9d3cd0678c50292a
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


class PrimitiveOp_12c46b768fa67cb4191c88f966b666ff(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.flatten(input_0, 0, 1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 300, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_2f8d200b02c208b527c70395d6dd6dc7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_12c46b768fa67cb4191c88f966b666ff
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


class TestPrimitiveOp_2c83db2f7440036ec58b2353678efcb5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_97c9dd9d0add9205a9f0d961ed6a22ee
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


class TestPrimitiveOp_6ab3ec6c37a40d5a1e58309b2f9790cd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_97c9dd9d0add9205a9f0d961ed6a22ee
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


class TestPrimitiveOp_acb9539c6703d849a35d13f3c4ad72f7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6d494b25d0d6cb134b2488a98f9ab641
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


class TestPrimitiveOp_5acf32bc91b8891d1a188c3350ae71b0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_97c9dd9d0add9205a9f0d961ed6a22ee
    def get_inputs(self):
        return [
            paddle.uniform([1, 68, 10, 10], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_832433196e96800de1a4d621a6d515c6(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.flatten(input_0, 2, 3), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[11, 768, 7, 7], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_d7a85992709b372e4b6fcb1ac51f5136(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_832433196e96800de1a4d621a6d515c6
    def get_inputs(self):
        return [
            paddle.uniform([11, 768, 7, 7], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_26ec071cf5c41c908773ecf9dfa8c17e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9f24a611edb72752eae6630f313d6010
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


class TestPrimitiveOp_2f19b56fb2361aa253d6cca4e33def8f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_97c9dd9d0add9205a9f0d961ed6a22ee
    def get_inputs(self):
        return [
            paddle.uniform([1, 68, 20, 20], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e70335f8251c9adb482c6d8f82d5d29d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7518a11010ee8b5ed8e90f35f7622dc7
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


class TestPrimitiveOp_4ac84e98943dad1063f5b5503d9f1283(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_97c9dd9d0add9205a9f0d961ed6a22ee
    def get_inputs(self):
        return [
            paddle.uniform([1, 68, 24, 24], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_09dec7a49e177afb7eb5ad1ea875bc1f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6d494b25d0d6cb134b2488a98f9ab641
    def get_inputs(self):
        return [
            paddle.uniform([1, 5, 16, 16], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_3c71c2cabfb63c368f8dce7075ebd59b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.flatten(input_0, 2, 3), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 384, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_21b2456e7853d08db9daf9de32707fd1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3c71c2cabfb63c368f8dce7075ebd59b
    def get_inputs(self):
        return [
            paddle.uniform([43, 384, 14, 14], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_08a43f87fe9d3c9522e5a122f83049d5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4a4a5d372a26e9b83fdaf9efbc66706f
    def get_inputs(self):
        return [
            paddle.uniform([1, 1280, 32, 64], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_fb2f2547f4150dc45b2bc0b24fc0db8d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.flatten(input_0, 2, 3), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[43, 768, 7, 7], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_53289b12faad3cbea6aaed15a6973a6b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fb2f2547f4150dc45b2bc0b24fc0db8d
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


class TestPrimitiveOp_f456d7b29f047c91d34e42e29ee54e42(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_97c9dd9d0add9205a9f0d961ed6a22ee
    def get_inputs(self):
        return [
            paddle.uniform([1, 68, 52, 52], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_007924ba9c584a2e0ae237b41ea856b1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bef188218c441198e62140fdb9816a3c
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


class TestPrimitiveOp_cd56cb6979fcf32a9e6ce2c42d24be74(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_97c9dd9d0add9205a9f0d961ed6a22ee
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


class TestPrimitiveOp_4d047ae5807f8da681bfdf80b93c51b8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7e725265c28a5e7a7e4f95d9a2ef1499
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


class TestPrimitiveOp_e7963c6cd558e4d76895cd669285f6df(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7e725265c28a5e7a7e4f95d9a2ef1499
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


class TestPrimitiveOp_bd1ee7279b491539e330b7f977bad933(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_97c9dd9d0add9205a9f0d961ed6a22ee
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


class TestPrimitiveOp_bd1ee7279b491539e330b7f977bad933(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_97c9dd9d0add9205a9f0d961ed6a22ee
    def get_inputs(self):
        return [
            paddle.uniform([1, 68, 28, 28], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e357ac02c591979ec5f8dcf86ca12acc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3c71c2cabfb63c368f8dce7075ebd59b
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


class TestPrimitiveOp_7d6c2bf4f41a6bb4e6e9f4ce63755b95(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_97c9dd9d0add9205a9f0d961ed6a22ee
    def get_inputs(self):
        return [
            paddle.uniform([1, 68, 38, 38], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_7380cbe9137a9a372a3aa8777dc593e8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2338c6ff1ed7583a883c5e3ac4a5d5a0
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


class TestPrimitiveOp_b9da1316d20b868e721ecc68a9d7bcff(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_97c9dd9d0add9205a9f0d961ed6a22ee
    def get_inputs(self):
        return [
            paddle.uniform([1, 68, 42, 42], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_fa5908fef298b3ff47464ce1e6a8f688(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.flatten(input_0, 2, 3), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[11, 192, 28, 28], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_20c1128cebe14b3717c35c6c4bd08831(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fa5908fef298b3ff47464ce1e6a8f688
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


class TestPrimitiveOp_603289838c5acc7bdf4c25f13de386e6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_97c9dd9d0add9205a9f0d961ed6a22ee
    def get_inputs(self):
        return [
            paddle.uniform([1, 68, 12, 12], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_6a7592ea66b83eb97a2a4cfffae5d733(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.flatten(input_0, 2, 3), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 512, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_34c594dd3b2b800180276fe8783e2109(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6a7592ea66b83eb97a2a4cfffae5d733
    def get_inputs(self):
        return [
            paddle.uniform([4, 512, 8, 32], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6d00b4935ddd893ca144f4907cdb7fd2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6a7592ea66b83eb97a2a4cfffae5d733
    def get_inputs(self):
        return [
            paddle.uniform([4, 512, 4, 32], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_180220f2afd7b6feadf5727f61cf6447(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6d494b25d0d6cb134b2488a98f9ab641
    def get_inputs(self):
        return [
            paddle.uniform([1, 5, 32, 32], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_acd6e357b4cc5f3b2ad8b1f1d921f1cb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_eb000f903e80fc402e871cadc1b93699
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


class TestPrimitiveOp_b074976952ecd4283aa6319962c41066(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_97c9dd9d0add9205a9f0d961ed6a22ee
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


class TestPrimitiveOp_762007e08255115eae978e3a29ae4335(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9f24a611edb72752eae6630f313d6010
    def get_inputs(self):
        return [
            paddle.uniform([11, 96, 56, 56], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_9d9129696e1f60bf7d69ab54fb24cd41(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.flatten(input_0, 2, 3), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[43, 96, 56, 56], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_8ebd8eb3ee793e6baeece9271d69796f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9d9129696e1f60bf7d69ab54fb24cd41
    def get_inputs(self):
        return [
            paddle.uniform([43, 96, 56, 56], dtype='float32', min=0, max=0.5),
        ]




if __name__ == '__main__':
    unittest.main()