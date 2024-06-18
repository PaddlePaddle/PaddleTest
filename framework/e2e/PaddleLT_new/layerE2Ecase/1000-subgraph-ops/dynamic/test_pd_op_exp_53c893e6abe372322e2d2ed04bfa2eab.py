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



class PrimitiveOp_a54e8b7fd430ccae91eb6ccbd5842361(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.exp(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_a17362c42cab420fac265ba6a70d9381(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a54e8b7fd430ccae91eb6ccbd5842361
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=-0.5, max=0.5),
        ]


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


class TestPrimitiveOp_9b1ff32ed3a51e454f37c2e52acab808(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ff26e98f4783c5fb3b5da69149dd25f5
    def get_inputs(self):
        return [
            paddle.uniform([1, 6, 21824], dtype='float32', min=-0.5, max=0.5),
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


class TestPrimitiveOp_653c12f2aee30f12166a15642e61efde(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b4ecbf49e01c987d4a1f4b7826bc1e40
    def get_inputs(self):
        return [
            paddle.uniform([10, 4, 100, 100], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_9d7f11a91c99c865a233e476243b2895(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b4ecbf49e01c987d4a1f4b7826bc1e40
    def get_inputs(self):
        return [
            paddle.uniform([1, 12, 1025, 1025], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_00d7b015e832b18e2642e7c7b404ecc5(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.exp(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_096ef2f2dc3be97b1baf7108b68d87af(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_00d7b015e832b18e2642e7c7b404ecc5
    def get_inputs(self):
        return [
            paddle.uniform([15200, 80], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_647e730d492f45dbab94443829feaa13(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b4ecbf49e01c987d4a1f4b7826bc1e40
    def get_inputs(self):
        return [
            paddle.uniform([1, 8, 1024, 1024], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_1f7caaf2d4b6db6b6f99307af8e69f95(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b4ecbf49e01c987d4a1f4b7826bc1e40
    def get_inputs(self):
        return [
            paddle.uniform([54, 3, 197, 197], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_358a25cd7cf492da9f447a0229662108(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ff26e98f4783c5fb3b5da69149dd25f5
    def get_inputs(self):
        return [
            paddle.to_tensor([[[0.09411972761154175]], [[-0.05412188172340393]], [[-0.10265585780143738]], [[0.2077065110206604]], [[0.022561192512512207]], [[0.3905274271965027]]], dtype='float32').reshape([6, 1, 1]),
        ]


class TestPrimitiveOp_b9d58f0738bbef29566ea80d51eaea54(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ff26e98f4783c5fb3b5da69149dd25f5
    def get_inputs(self):
        return [
            paddle.to_tensor([[[0.14321905374526978]], [[0.002419412136077881]], [[0.3457615375518799]], [[-0.48177680373191833]], [[-0.29202499985694885]], [[0.3379940390586853]]], dtype='float32').reshape([6, 1, 1]),
        ]


class TestPrimitiveOp_71a5f6e566c5d70f33e08e1002d7ae4a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b4ecbf49e01c987d4a1f4b7826bc1e40
    def get_inputs(self):
        return [
            paddle.uniform([1, 7581, 4, 17], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_ca07ea834020891439b84d76f496fc2d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b4ecbf49e01c987d4a1f4b7826bc1e40
    def get_inputs(self):
        return [
            paddle.uniform([1, 2100, 4, 17], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_d393858ccf93ce583eeed1ad253f650d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a54e8b7fd430ccae91eb6ccbd5842361
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 17, 50, 76], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_cb4411db0178613618b1bfb7aa803e88(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b4ecbf49e01c987d4a1f4b7826bc1e40
    def get_inputs(self):
        return [
            paddle.uniform([10, 2, 640, 640], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_e5d9a75f5fbc6344d151e420458cb564(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a54e8b7fd430ccae91eb6ccbd5842361
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_a558096cb21f61ee4dee7f55d9341ce0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a54e8b7fd430ccae91eb6ccbd5842361
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 17, 100, 152], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_35b5820772102b78ca078b269b56ef54(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ff26e98f4783c5fb3b5da69149dd25f5
    def get_inputs(self):
        return [
            paddle.uniform([1, 21, 16384], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_135a9a4ba1fb2343480025d63fc1b697(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b4ecbf49e01c987d4a1f4b7826bc1e40
    def get_inputs(self):
        return [
            paddle.uniform([10, 2, 200, 200], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_91e33fa0be30b7cd2ebc7bcd3a6c3734(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b4ecbf49e01c987d4a1f4b7826bc1e40
    def get_inputs(self):
        return [
            paddle.uniform([1, 9261, 4, 17], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_826eb8936e34c384432761b39098c225(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b4ecbf49e01c987d4a1f4b7826bc1e40
    def get_inputs(self):
        return [
            paddle.uniform([1, 4116, 4, 17], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_85f0268dcd55a297a89872faab250bd9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_00d7b015e832b18e2642e7c7b404ecc5
    def get_inputs(self):
        return [
            paddle.uniform([3800, 80], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_b29e6f5a8f344901ba008cc43c0e4cc0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a54e8b7fd430ccae91eb6ccbd5842361
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 17, 7, 10], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_efb9b23d05a82d1b3a8ce954bf0cbb1c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ff26e98f4783c5fb3b5da69149dd25f5
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 512], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_9c5de3cee65772171b807d565a60d11c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ff26e98f4783c5fb3b5da69149dd25f5
    def get_inputs(self):
        return [
            paddle.uniform([1, 21504, 2], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_7d9256e7486b12905095a77385e0c615(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b4ecbf49e01c987d4a1f4b7826bc1e40
    def get_inputs(self):
        return [
            paddle.uniform([1, 21504, 1, 91], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_85f0268dcd55a297a89872faab250bd9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_00d7b015e832b18e2642e7c7b404ecc5
    def get_inputs(self):
        return [
            paddle.uniform([3800, 80], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_cf3c0484b0dbf3264ac8e313b3721b0e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b4ecbf49e01c987d4a1f4b7826bc1e40
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 128, 128], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_961d2c546ac2dafc2c050f7bd5aea52b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_00d7b015e832b18e2642e7c7b404ecc5
    def get_inputs(self):
        return [
            paddle.uniform([2204, 80], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_6f4705b98bcd89ddd36bd97d9262a101(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b4ecbf49e01c987d4a1f4b7826bc1e40
    def get_inputs(self):
        return [
            paddle.uniform([10, 8, 160, 160], dtype='float32', min=-0.5, max=0.5),
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


class TestPrimitiveOp_f9e0dc0271a8902b42cbb6cb6b52b62a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4e8cab16b6a7ae943ee5a3e23e1d4e0e
    def get_inputs(self):
        return [
            paddle.to_tensor([0.13141000270843506], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_b862c66a3159c2c18b162eb628e7972e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4e8cab16b6a7ae943ee5a3e23e1d4e0e
    def get_inputs(self):
        return [
            paddle.to_tensor([0.2926388680934906], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_83493e8650ebc09b4a328d9329f350cf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a54e8b7fd430ccae91eb6ccbd5842361
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_3b228eb5261de6ccc6eea66522897b14(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4e8cab16b6a7ae943ee5a3e23e1d4e0e
    def get_inputs(self):
        return [
            paddle.uniform([256], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_866ffc8343ef6632634ba8b067fd4d5c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b4ecbf49e01c987d4a1f4b7826bc1e40
    def get_inputs(self):
        return [
            paddle.uniform([10, 4, 320, 320], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_98b1cc9e34802c4283698a3bf656b2d3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_00d7b015e832b18e2642e7c7b404ecc5
    def get_inputs(self):
        return [
            paddle.uniform([950, 80], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_f38196ee552ea8b2e587137cbc3d2517(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a54e8b7fd430ccae91eb6ccbd5842361
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_db3555cc12448ad2d8b15975fd857ccf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b4ecbf49e01c987d4a1f4b7826bc1e40
    def get_inputs(self):
        return [
            paddle.uniform([2, 1, 960, 960], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_351c7de8a21c662932ce0d03a23b4ffe(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ff26e98f4783c5fb3b5da69149dd25f5
    def get_inputs(self):
        return [
            paddle.uniform([1, 8400, 2], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_5de0498e2d11758c0286742a9794b2c7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b4ecbf49e01c987d4a1f4b7826bc1e40
    def get_inputs(self):
        return [
            paddle.uniform([1, 8, 512, 512], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_269e19b6322469cc119793316684a2b3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b4ecbf49e01c987d4a1f4b7826bc1e40
    def get_inputs(self):
        return [
            paddle.uniform([54, 3, 198, 198], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_7f80643c6067edc56ecce181a0c2ad48(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b4ecbf49e01c987d4a1f4b7826bc1e40
    def get_inputs(self):
        return [
            paddle.uniform([1, 3549, 4, 17], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_647e730d492f45dbab94443829feaa13(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b4ecbf49e01c987d4a1f4b7826bc1e40
    def get_inputs(self):
        return [
            paddle.uniform([1, 8, 1024, 1024], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_f20a1c3b75b43120f0c267828bb2d420(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b4ecbf49e01c987d4a1f4b7826bc1e40
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 16, 16], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_57fa0519882d3e7f6b1ec0dfd0746882(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_00d7b015e832b18e2642e7c7b404ecc5
    def get_inputs(self):
        return [
            paddle.uniform([8816, 80], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_2bd242b916145f83ba859a5e1eda3455(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b4ecbf49e01c987d4a1f4b7826bc1e40
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 64, 64], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_b200d93a7f8ebb89b1064dbedb917aa7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ff26e98f4783c5fb3b5da69149dd25f5
    def get_inputs(self):
        return [
            paddle.uniform([1, 19, 32768], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_201176c857e1e09ded8160b0518eabea(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b4ecbf49e01c987d4a1f4b7826bc1e40
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 32, 32], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_ae8b11460920134881b7a544490d9464(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ff26e98f4783c5fb3b5da69149dd25f5
    def get_inputs(self):
        return [
            paddle.to_tensor([[[-0.3317444920539856]], [[0.4688023328781128]], [[-0.4925745129585266]], [[0.30466246604919434]], [[0.21355074644088745]], [[-0.296505331993103]]], dtype='float32').reshape([6, 1, 1]),
        ]


class TestPrimitiveOp_19147874ce25fa101514e76157a67fe6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ff26e98f4783c5fb3b5da69149dd25f5
    def get_inputs(self):
        return [
            paddle.to_tensor([[[-0.03960481286048889]], [[-0.033424556255340576]], [[-0.10159653425216675]], [[0.26393401622772217]], [[-0.4051783084869385]], [[-0.038964927196502686]]], dtype='float32').reshape([6, 1, 1]),
        ]


class TestPrimitiveOp_9f064609137b92f6ea6fe7a8a4647cd6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a54e8b7fd430ccae91eb6ccbd5842361
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_6d35f608d63194b995db44e6bf22b0a8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a54e8b7fd430ccae91eb6ccbd5842361
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_c8525233da649642ab38117749fe5df1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_00d7b015e832b18e2642e7c7b404ecc5
    def get_inputs(self):
        return [
            paddle.uniform([150, 80], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_f22392c27290a8b4f6057861f39ea40d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b4ecbf49e01c987d4a1f4b7826bc1e40
    def get_inputs(self):
        return [
            paddle.uniform([1, 12, 1174, 1174], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_e3d74b00e7695ae1ea8de3c8704cc2cd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b4ecbf49e01c987d4a1f4b7826bc1e40
    def get_inputs(self):
        return [
            paddle.uniform([1, 4725, 4, 17], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_ddeb40575727ec7e7e67763fb05597ae(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b4ecbf49e01c987d4a1f4b7826bc1e40
    def get_inputs(self):
        return [
            paddle.uniform([1, 11109, 4, 17], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_7e3662865301fd81674217ed512dab0b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b4ecbf49e01c987d4a1f4b7826bc1e40
    def get_inputs(self):
        return [
            paddle.uniform([86, 3, 198, 198], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_fe72a6d855c859bbe3e298055521dbf1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_00d7b015e832b18e2642e7c7b404ecc5
    def get_inputs(self):
        return [
            paddle.uniform([70, 80], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_d3116228b722aa6af7366174fad27f9f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ff26e98f4783c5fb3b5da69149dd25f5
    def get_inputs(self):
        return [
            paddle.uniform([1, 12096, 2], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_a950e8969230f98b98b9e137fd8abb71(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a54e8b7fd430ccae91eb6ccbd5842361
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_f67c45d576362d111fe7cdffd6979b6a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a54e8b7fd430ccae91eb6ccbd5842361
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_f660e1e1d1d512fe5c4cf61ae301518b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ff26e98f4783c5fb3b5da69149dd25f5
    def get_inputs(self):
        return [
            paddle.uniform([1, 8192, 8192], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_355f6ae9592c9aa149484b402110b724(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b4ecbf49e01c987d4a1f4b7826bc1e40
    def get_inputs(self):
        return [
            paddle.uniform([10, 8, 50, 50], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_3403415e6ce02178e22f04224dec6542(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ff26e98f4783c5fb3b5da69149dd25f5
    def get_inputs(self):
        return [
            paddle.uniform([1, 6069, 2], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_a8c7633a0d68855d1c6f29e619e7bc97(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b4ecbf49e01c987d4a1f4b7826bc1e40
    def get_inputs(self):
        return [
            paddle.uniform([1, 6, 1025, 1025], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_be8363005a4c803c4f3ba19bdf8d103e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b4ecbf49e01c987d4a1f4b7826bc1e40
    def get_inputs(self):
        return [
            paddle.uniform([1, 8400, 4, 17], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_a169dbd14d842a8caf7bdf28a4aebecf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a54e8b7fd430ccae91eb6ccbd5842361
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_5e16530236d962b2d751299d869311f3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b4ecbf49e01c987d4a1f4b7826bc1e40
    def get_inputs(self):
        return [
            paddle.uniform([86, 3, 197, 197], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_3b228eb5261de6ccc6eea66522897b14(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4e8cab16b6a7ae943ee5a3e23e1d4e0e
    def get_inputs(self):
        return [
            paddle.uniform([256], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_a88338759782ceb90549aa1b444f7979(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a54e8b7fd430ccae91eb6ccbd5842361
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 17, 13, 19], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_7d3137c9318539274a35a1c207105348(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a54e8b7fd430ccae91eb6ccbd5842361
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_1e52eb3e162cdb9fa2c1a71d30e30fd6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ff26e98f4783c5fb3b5da69149dd25f5
    def get_inputs(self):
        return [
            paddle.uniform([1, 4096, 4096], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_fa075a4d2066cc98dbb94ff62beb6b56(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a54e8b7fd430ccae91eb6ccbd5842361
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_b63882be7d5cca75c7f31c403ad21f1a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b4ecbf49e01c987d4a1f4b7826bc1e40
    def get_inputs(self):
        return [
            paddle.uniform([1, 12, 577, 577], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_efb9b23d05a82d1b3a8ce954bf0cbb1c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ff26e98f4783c5fb3b5da69149dd25f5
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 512], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_cc4223072202a7535a7ea8f312d29418(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a54e8b7fd430ccae91eb6ccbd5842361
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 17, 25, 38], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_657f9d97f7d864e857762db059a7a34b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4e8cab16b6a7ae943ee5a3e23e1d4e0e
    def get_inputs(self):
        return [
            paddle.to_tensor([0.07609263062477112], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_5bfa42316c8df760f48afde871ae1554(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4e8cab16b6a7ae943ee5a3e23e1d4e0e
    def get_inputs(self):
        return [
            paddle.to_tensor([0.08249622583389282], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_bb92fd1bf49ef1570d98a25eff84a620(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4e8cab16b6a7ae943ee5a3e23e1d4e0e
    def get_inputs(self):
        return [
            paddle.to_tensor([0.3798350691795349], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_ae7b6612cd3e5d2f3a2d2b74177c8678(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4e8cab16b6a7ae943ee5a3e23e1d4e0e
    def get_inputs(self):
        return [
            paddle.to_tensor([0.001259535551071167], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_b82f6603b9a2d8b993aa1c1d6d9ccc89(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4e8cab16b6a7ae943ee5a3e23e1d4e0e
    def get_inputs(self):
        return [
            paddle.to_tensor([0.43146899342536926], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_88910c5d1c5607271c43b43b3ddc1b94(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4e8cab16b6a7ae943ee5a3e23e1d4e0e
    def get_inputs(self):
        return [
            paddle.to_tensor([-0.4421810507774353], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_7cab3970b2d547b395ca50abf8b13955(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4e8cab16b6a7ae943ee5a3e23e1d4e0e
    def get_inputs(self):
        return [
            paddle.to_tensor([0.3128872513771057], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_f9371678a092d230515ff22648237001(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4e8cab16b6a7ae943ee5a3e23e1d4e0e
    def get_inputs(self):
        return [
            paddle.to_tensor([0.17677366733551025], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_112d58d14f1c7b5866b13338ee85896c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4e8cab16b6a7ae943ee5a3e23e1d4e0e
    def get_inputs(self):
        return [
            paddle.to_tensor([-0.4452681541442871], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_0f7314efa827b35bc01d4cff11ca7ccc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b4ecbf49e01c987d4a1f4b7826bc1e40
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 8, 8], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_5bb3f17184b5acf7366edb98d70f562b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a54e8b7fd430ccae91eb6ccbd5842361
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_299161eb6efd1a2d37ba041b2cd595e0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a54e8b7fd430ccae91eb6ccbd5842361
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_f4e008f988e7f2c0720eff010f638e53(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a54e8b7fd430ccae91eb6ccbd5842361
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_8a73a0d026852925abf9309aa138ffb2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_00d7b015e832b18e2642e7c7b404ecc5
    def get_inputs(self):
        return [
            paddle.uniform([551, 80], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_4e74d22db3dc2d8cd96b58833e2a8617(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ff26e98f4783c5fb3b5da69149dd25f5
    def get_inputs(self):
        return [
            paddle.uniform([1, 6804, 2], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_af31e1ddc1aff04205a7239cfe4af0e5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ff26e98f4783c5fb3b5da69149dd25f5
    def get_inputs(self):
        return [
            paddle.uniform([1, 5376, 2], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_866ffc8343ef6632634ba8b067fd4d5c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b4ecbf49e01c987d4a1f4b7826bc1e40
    def get_inputs(self):
        return [
            paddle.uniform([10, 4, 320, 320], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_5de0498e2d11758c0286742a9794b2c7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b4ecbf49e01c987d4a1f4b7826bc1e40
    def get_inputs(self):
        return [
            paddle.uniform([1, 8, 512, 512], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_653c12f2aee30f12166a15642e61efde(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b4ecbf49e01c987d4a1f4b7826bc1e40
    def get_inputs(self):
        return [
            paddle.uniform([10, 4, 100, 100], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_cd0738807d9797e6511601fbff6b4207(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b4ecbf49e01c987d4a1f4b7826bc1e40
    def get_inputs(self):
        return [
            paddle.uniform([1, 6, 1174, 1174], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_990e0a9e48bb2281fff1567272a674d0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a54e8b7fd430ccae91eb6ccbd5842361
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_58cb46537e046f91c53b1164273465c3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b4ecbf49e01c987d4a1f4b7826bc1e40
    def get_inputs(self):
        return [
            paddle.uniform([1, 6069, 4, 17], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_fabde46a0a31144fd2b870b26e6a336d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b4ecbf49e01c987d4a1f4b7826bc1e40
    def get_inputs(self):
        return [
            paddle.uniform([1, 3549, 4, 19], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_9ab2e9ba4cb44490b5cedb7301942e4f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b4ecbf49e01c987d4a1f4b7826bc1e40
    def get_inputs(self):
        return [
            paddle.uniform([1, 3024, 4, 17], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_faa0e002f7530cd9b45cd66017bf8457(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_00d7b015e832b18e2642e7c7b404ecc5
    def get_inputs(self):
        return [
            paddle.uniform([247, 80], dtype='float32', min=-0.5, max=0.5),
        ]




if __name__ == '__main__':
    unittest.main()