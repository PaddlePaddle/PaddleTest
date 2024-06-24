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


class TestPrimitiveOp_38040225fc9e2889a7ef10e79710b26a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ff26e98f4783c5fb3b5da69149dd25f5
    def get_inputs(self):
        return [
            paddle.to_tensor([[[0.32510143518447876]], [[0.07775720953941345]], [[0.28316885232925415]], [[0.31901025772094727]], [[0.4317087233066559]], [[0.15360833704471588]]], dtype='float32').reshape([6, 1, 1]),
        ]


class TestPrimitiveOp_ab5a5f8609de61d5c615d0f2ff45a277(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ff26e98f4783c5fb3b5da69149dd25f5
    def get_inputs(self):
        return [
            paddle.to_tensor([[[0.2087109237909317]], [[0.22698462009429932]], [[0.39309483766555786]], [[0.44628360867500305]], [[0.010610256344079971]], [[0.23917493224143982]]], dtype='float32').reshape([6, 1, 1]),
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


class TestPrimitiveOp_13efcb7509130372ac827de61929f3f7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ff26e98f4783c5fb3b5da69149dd25f5
    def get_inputs(self):
        return [
            paddle.to_tensor([[[0.34319785237312317]], [[0.1778753101825714]], [[0.014414490200579166]], [[0.4434035122394562]], [[0.4677691161632538]], [[0.19076691567897797]]], dtype='float32').reshape([6, 1, 1]),
        ]


class TestPrimitiveOp_bc962928828a3c2b9c2a899a62104069(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ff26e98f4783c5fb3b5da69149dd25f5
    def get_inputs(self):
        return [
            paddle.to_tensor([[[0.07857821136713028]], [[0.407955139875412]], [[0.08358895033597946]], [[0.15103097259998322]], [[0.47805821895599365]], [[0.07587911933660507]]], dtype='float32').reshape([6, 1, 1]),
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


class TestPrimitiveOp_4577f6d8afaa12702912d44100a7c366(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4e8cab16b6a7ae943ee5a3e23e1d4e0e
    def get_inputs(self):
        return [
            paddle.to_tensor([-0.4762192368507385], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_6166a995baf7a9eecc8b7d4ce9a1f77e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4e8cab16b6a7ae943ee5a3e23e1d4e0e
    def get_inputs(self):
        return [
            paddle.to_tensor([-0.06927169859409332], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_dfa3d157eb6624bdde73cc2f9e6f2aad(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ff26e98f4783c5fb3b5da69149dd25f5
    def get_inputs(self):
        return [
            paddle.uniform([1, 8400, 2], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e3aa34ab911ad4820f8a9860fac6e94a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4e8cab16b6a7ae943ee5a3e23e1d4e0e
    def get_inputs(self):
        return [
            paddle.to_tensor([-0.1993337869644165], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_f9fababee7682a85fbf242cd70e8f4b2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4e8cab16b6a7ae943ee5a3e23e1d4e0e
    def get_inputs(self):
        return [
            paddle.to_tensor([-0.09994659572839737], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_6dd48224322a31a08684fd8aea5b5a07(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4e8cab16b6a7ae943ee5a3e23e1d4e0e
    def get_inputs(self):
        return [
            paddle.to_tensor([-0.4568828344345093], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_23b7ff6b1becd89b68154c92822ed121(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4e8cab16b6a7ae943ee5a3e23e1d4e0e
    def get_inputs(self):
        return [
            paddle.to_tensor([-0.31034454703330994], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_bf8cb328321153de8ef04dd4f7a243f6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4e8cab16b6a7ae943ee5a3e23e1d4e0e
    def get_inputs(self):
        return [
            paddle.to_tensor([-0.06750066578388214], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_ca9a19f338658c2a7b59c8ad0b437020(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4e8cab16b6a7ae943ee5a3e23e1d4e0e
    def get_inputs(self):
        return [
            paddle.to_tensor([-0.16773703694343567], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_aa7dfdf703b989980dce7ae455e849c9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4e8cab16b6a7ae943ee5a3e23e1d4e0e
    def get_inputs(self):
        return [
            paddle.to_tensor([-0.15381911396980286], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_0935212a718be09eb2717e6c87fa143b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4e8cab16b6a7ae943ee5a3e23e1d4e0e
    def get_inputs(self):
        return [
            paddle.to_tensor([-0.3341449797153473], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_c36a5b0a546da51de76ab1e5e0fba341(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4e8cab16b6a7ae943ee5a3e23e1d4e0e
    def get_inputs(self):
        return [
            paddle.to_tensor([-0.18431654572486877], dtype='float32').reshape([1]),
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