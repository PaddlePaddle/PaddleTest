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



class PrimitiveOp_bb177be340c9c58e2e08e7b48f772261(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.expand(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None], dtype='float32'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_70975347f19f4a6357563d870bee4439(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bb177be340c9c58e2e08e7b48f772261
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([1, 3, 23, 23, 1], dtype='int64').reshape([5]),
        ]


class TestPrimitiveOp_a49682bf2589be3a70802c9ca6d607be(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bb177be340c9c58e2e08e7b48f772261
    def get_inputs(self):
        return [
            paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            paddle.to_tensor([1, 3, 23, 23, 1], dtype='int64').reshape([5]),
        ]


class TestPrimitiveOp_3e4d77081a3e56cdacd55f36cfadb947(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bb177be340c9c58e2e08e7b48f772261
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([22, 60], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_65abaf12aa1d89498a602e179ee28711(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bb177be340c9c58e2e08e7b48f772261
    def get_inputs(self):
        return [
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            paddle.to_tensor([22, 60], dtype='int64').reshape([2]),
        ]


class PrimitiveOp_635d62ccba7f178d0d1030f27e02e8bb(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.expand(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None], dtype='float32'),
            paddle.static.InputSpec(shape=[None], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_9b2804a56be1e8d8baf0ad5f879ecf52(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_635d62ccba7f178d0d1030f27e02e8bb
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([1524, 1], dtype='int32').reshape([2]),
        ]


class TestPrimitiveOp_fe5a0312cca517af15cc214fe6989568(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_635d62ccba7f178d0d1030f27e02e8bb
    def get_inputs(self):
        return [
            paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            paddle.to_tensor([1524, 1], dtype='int32').reshape([2]),
        ]


class TestPrimitiveOp_9b2804a56be1e8d8baf0ad5f879ecf52(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_635d62ccba7f178d0d1030f27e02e8bb
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([1524, 1], dtype='int32').reshape([2]),
        ]


class TestPrimitiveOp_fe5a0312cca517af15cc214fe6989568(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_635d62ccba7f178d0d1030f27e02e8bb
    def get_inputs(self):
        return [
            paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            paddle.to_tensor([1524, 1], dtype='int32').reshape([2]),
        ]


class TestPrimitiveOp_d0e76d984511afd49d00535e04b3dbf8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bb177be340c9c58e2e08e7b48f772261
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([1, 3024, 4], dtype='int64').reshape([3]),
        ]


class TestPrimitiveOp_3bbb1c658246413615d3bbe4f25a8802(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bb177be340c9c58e2e08e7b48f772261
    def get_inputs(self):
        return [
            paddle.to_tensor([15.989999771118164], dtype='float32').reshape([1]),
            paddle.to_tensor([1, 3024, 4], dtype='int64').reshape([3]),
        ]


class TestPrimitiveOp_6e63e3f17aa0821925628734c1601091(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bb177be340c9c58e2e08e7b48f772261
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([1, 672], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_f8e614e5a662b959525f55d3c2c21b9e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bb177be340c9c58e2e08e7b48f772261
    def get_inputs(self):
        return [
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            paddle.to_tensor([1, 672], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_7d51671423c8c35b34bf8f602b3f7590(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_635d62ccba7f178d0d1030f27e02e8bb
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([2340, 1], dtype='int32').reshape([2]),
        ]


class TestPrimitiveOp_c7041c07beb8fd05864fe669ef3393c5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_635d62ccba7f178d0d1030f27e02e8bb
    def get_inputs(self):
        return [
            paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            paddle.to_tensor([2340, 1], dtype='int32').reshape([2]),
        ]


class TestPrimitiveOp_7d51671423c8c35b34bf8f602b3f7590(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_635d62ccba7f178d0d1030f27e02e8bb
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([2340, 1], dtype='int32').reshape([2]),
        ]


class TestPrimitiveOp_c7041c07beb8fd05864fe669ef3393c5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_635d62ccba7f178d0d1030f27e02e8bb
    def get_inputs(self):
        return [
            paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            paddle.to_tensor([2340, 1], dtype='int32').reshape([2]),
        ]


class TestPrimitiveOp_9d07703efe932a4062d28878b8dd2eef(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bb177be340c9c58e2e08e7b48f772261
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([1, 4725, 4], dtype='int64').reshape([3]),
        ]


class TestPrimitiveOp_3df80e560cf2e2f9bc86da7f3047a08c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bb177be340c9c58e2e08e7b48f772261
    def get_inputs(self):
        return [
            paddle.to_tensor([15.989999771118164], dtype='float32').reshape([1]),
            paddle.to_tensor([1, 4725, 4], dtype='int64').reshape([3]),
        ]


class TestPrimitiveOp_fcf48fbf5bcdd96f52d200311b619415(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bb177be340c9c58e2e08e7b48f772261
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([171, 480], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_ea6b2bbfaa7be7d54fd16e1144e801c2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bb177be340c9c58e2e08e7b48f772261
    def get_inputs(self):
        return [
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            paddle.to_tensor([171, 480], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_5d469bdc862f8880608985ddb98f1b24(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bb177be340c9c58e2e08e7b48f772261
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([1, 3, 12, 12, 1], dtype='int64').reshape([5]),
        ]


class TestPrimitiveOp_3528cf6bba64dca0d91fe74c3a3ec476(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bb177be340c9c58e2e08e7b48f772261
    def get_inputs(self):
        return [
            paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            paddle.to_tensor([1, 3, 12, 12, 1], dtype='int64').reshape([5]),
        ]


class TestPrimitiveOp_0d89acd64472b0f108895fa67eaff040(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bb177be340c9c58e2e08e7b48f772261
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([145, 60], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_8cb8716e564e3bb00ed73e587c55444c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bb177be340c9c58e2e08e7b48f772261
    def get_inputs(self):
        return [
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            paddle.to_tensor([145, 60], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_71a4c8780787e996943c88d28ae589a3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bb177be340c9c58e2e08e7b48f772261
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([4, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_796845375f725d0bbd9551f882b0fabf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bb177be340c9c58e2e08e7b48f772261
    def get_inputs(self):
        return [
            paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            paddle.to_tensor([4, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_0af6ef3e89db771c2bf03932db126729(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bb177be340c9c58e2e08e7b48f772261
    def get_inputs(self):
        return [
            paddle.to_tensor([-10000000000.0], dtype='float32').reshape([1]),
            paddle.to_tensor([6, 1, 1], dtype='int64').reshape([3]),
        ]


class TestPrimitiveOp_0294de3f996c1c86f328d93f231fd00e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bb177be340c9c58e2e08e7b48f772261
    def get_inputs(self):
        return [
            paddle.to_tensor([4.135169982910156], dtype='float32').reshape([1]),
            paddle.to_tensor([6, 1, 1], dtype='int64').reshape([3]),
        ]


class TestPrimitiveOp_8ac7157f025dc7f91424274e20b7f8c9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bb177be340c9c58e2e08e7b48f772261
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([145, 336], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_513330871590bae936e1ce0eadde0c63(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bb177be340c9c58e2e08e7b48f772261
    def get_inputs(self):
        return [
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            paddle.to_tensor([145, 336], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_ad1afa736c44ac02cb4677fc477434f5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bb177be340c9c58e2e08e7b48f772261
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([1, 3, 46, 46, 1], dtype='int64').reshape([5]),
        ]


class TestPrimitiveOp_fb6e0a778cc9826de9372cdaabc63715(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bb177be340c9c58e2e08e7b48f772261
    def get_inputs(self):
        return [
            paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            paddle.to_tensor([1, 3, 46, 46, 1], dtype='int64').reshape([5]),
        ]


class TestPrimitiveOp_9683024a38d7e36ea193edd657310930(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bb177be340c9c58e2e08e7b48f772261
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([1, 3, 76, 76, 1], dtype='int64').reshape([5]),
        ]


class TestPrimitiveOp_4f9bb068dd49253749689edbc1234eff(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bb177be340c9c58e2e08e7b48f772261
    def get_inputs(self):
        return [
            paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            paddle.to_tensor([1, 3, 76, 76, 1], dtype='int64').reshape([5]),
        ]


class TestPrimitiveOp_856ac5aa9cab122cd8c71822784b8982(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bb177be340c9c58e2e08e7b48f772261
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([22, 240], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_ff8ce7a176110cab81ff74b156924c5b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bb177be340c9c58e2e08e7b48f772261
    def get_inputs(self):
        return [
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            paddle.to_tensor([22, 240], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_71903354d6dcfe7920698a8cdb1448db(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bb177be340c9c58e2e08e7b48f772261
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([1, 3, 92, 92, 1], dtype='int64').reshape([5]),
        ]


class TestPrimitiveOp_041977fb3ef2cc9cecb4043fdc03457e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bb177be340c9c58e2e08e7b48f772261
    def get_inputs(self):
        return [
            paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            paddle.to_tensor([1, 3, 92, 92, 1], dtype='int64').reshape([5]),
        ]


class TestPrimitiveOp_22fa97912fec343ca510ef767d47849e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bb177be340c9c58e2e08e7b48f772261
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([6, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_47737d37845e502827823c3cf591bbc1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bb177be340c9c58e2e08e7b48f772261
    def get_inputs(self):
        return [
            paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            paddle.to_tensor([6, 1], dtype='int64').reshape([2]),
        ]


class PrimitiveOp_8467b6650b538e5114f1e56ed72de4de(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.expand(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_8cfe78aac559141d820f13cb7f69dd6f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8467b6650b538e5114f1e56ed72de4de
    def get_inputs(self):
        return [
            paddle.uniform([1, 150, 384], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, -1, -1], dtype='int64').reshape([3]),
        ]


class TestPrimitiveOp_5da0ca461dfb41a634747fbd7125ccfd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bb177be340c9c58e2e08e7b48f772261
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([22, 336], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_50721740a542b08286df1babfeaebc7f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bb177be340c9c58e2e08e7b48f772261
    def get_inputs(self):
        return [
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            paddle.to_tensor([22, 336], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_e861b52fe0b74c5568eec154a4759b92(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bb177be340c9c58e2e08e7b48f772261
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([10, 60], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_af2008c099e7157438b2f751a884850c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bb177be340c9c58e2e08e7b48f772261
    def get_inputs(self):
        return [
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            paddle.to_tensor([10, 60], dtype='int64').reshape([2]),
        ]


class PrimitiveOp_ef22f064ba6b8bd11d8eacf9003d64b0(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.expand(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_b892d35826ae59bafbcc326ed0e4c9f1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ef22f064ba6b8bd11d8eacf9003d64b0
    def get_inputs(self):
        return [
            paddle.uniform([100, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([100, 152], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_b6d8aa551afd7cca1782295021df5c1e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ef22f064ba6b8bd11d8eacf9003d64b0
    def get_inputs(self):
        return [
            paddle.uniform([1, 152], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([100, 152], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_00f2c06a68f6cbac232a204e99e4b1c0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ef22f064ba6b8bd11d8eacf9003d64b0
    def get_inputs(self):
        return [
            paddle.uniform([50, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([50, 76], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_dff93394e4a56364853fafed6c6eb221(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ef22f064ba6b8bd11d8eacf9003d64b0
    def get_inputs(self):
        return [
            paddle.uniform([1, 76], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([50, 76], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_de9658b92baf08ab9eedae11e906eff2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ef22f064ba6b8bd11d8eacf9003d64b0
    def get_inputs(self):
        return [
            paddle.to_tensor([[16.0], [48.0], [80.0], [112.0], [144.0], [176.0], [208.0], [240.0], [272.0], [304.0], [336.0], [368.0], [400.0], [432.0], [464.0], [496.0], [528.0], [560.0], [592.0], [624.0], [656.0], [688.0], [720.0], [752.0], [784.0]], dtype='float32').reshape([25, 1]),
            paddle.to_tensor([25, 38], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_46cc225994ad01a410dd60fce1e8d534(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ef22f064ba6b8bd11d8eacf9003d64b0
    def get_inputs(self):
        return [
            paddle.uniform([1, 38], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([25, 38], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_f3a1271bc2e5bd5c4cd31e0ad54f3c7f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ef22f064ba6b8bd11d8eacf9003d64b0
    def get_inputs(self):
        return [
            paddle.to_tensor([[32.0], [96.0], [160.0], [224.0], [288.0], [352.0], [416.0], [480.0], [544.0], [608.0], [672.0], [736.0], [800.0]], dtype='float32').reshape([13, 1]),
            paddle.to_tensor([13, 19], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_8e8c6eecd36eed7dacb1a636bfd68364(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ef22f064ba6b8bd11d8eacf9003d64b0
    def get_inputs(self):
        return [
            paddle.to_tensor([[32.0, 96.0, 160.0, 224.0, 288.0, 352.0, 416.0, 480.0, 544.0, 608.0, 672.0, 736.0, 800.0, 864.0, 928.0, 992.0, 1056.0, 1120.0, 1184.0]], dtype='float32').reshape([1, 19]),
            paddle.to_tensor([13, 19], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_2d370883c0a8d0da42c2b25a6dc169b1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ef22f064ba6b8bd11d8eacf9003d64b0
    def get_inputs(self):
        return [
            paddle.to_tensor([[64.0], [192.0], [320.0], [448.0], [576.0], [704.0], [832.0]], dtype='float32').reshape([7, 1]),
            paddle.to_tensor([7, 10], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_9d62a542f9afe2f20149de0a26763527(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ef22f064ba6b8bd11d8eacf9003d64b0
    def get_inputs(self):
        return [
            paddle.to_tensor([[64.0, 192.0, 320.0, 448.0, 576.0, 704.0, 832.0, 960.0, 1088.0, 1216.0]], dtype='float32').reshape([1, 10]),
            paddle.to_tensor([7, 10], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_ac9cce63a5b54e059788fd4341c8f672(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bb177be340c9c58e2e08e7b48f772261
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([1, 3, 24, 24, 1], dtype='int64').reshape([5]),
        ]


class TestPrimitiveOp_0efefb3dbdfbaec909645753af2263ce(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bb177be340c9c58e2e08e7b48f772261
    def get_inputs(self):
        return [
            paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            paddle.to_tensor([1, 3, 24, 24, 1], dtype='int64').reshape([5]),
        ]


class TestPrimitiveOp_401c129ad3163710b57889025fa799d9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bb177be340c9c58e2e08e7b48f772261
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([1, 960], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_265a80f04743087e616145edcf78451a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bb177be340c9c58e2e08e7b48f772261
    def get_inputs(self):
        return [
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            paddle.to_tensor([1, 960], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_44eb28e6938baf3f769dec71b9e57b08(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bb177be340c9c58e2e08e7b48f772261
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([1, 3, 19, 19, 1], dtype='int64').reshape([5]),
        ]


class TestPrimitiveOp_faf3c4a9f0c942a4b2156c5dc76cab18(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bb177be340c9c58e2e08e7b48f772261
    def get_inputs(self):
        return [
            paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            paddle.to_tensor([1, 3, 19, 19, 1], dtype='int64').reshape([5]),
        ]


class TestPrimitiveOp_bf8f07c6b0601510db37d3a170922b6c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_635d62ccba7f178d0d1030f27e02e8bb
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([2047, 1], dtype='int32').reshape([2]),
        ]


class TestPrimitiveOp_78c5e01cb1322f4dcf8942fe1cc6fb4e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_635d62ccba7f178d0d1030f27e02e8bb
    def get_inputs(self):
        return [
            paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            paddle.to_tensor([2047, 1], dtype='int32').reshape([2]),
        ]


class TestPrimitiveOp_bf8f07c6b0601510db37d3a170922b6c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_635d62ccba7f178d0d1030f27e02e8bb
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([2047, 1], dtype='int32').reshape([2]),
        ]


class TestPrimitiveOp_78c5e01cb1322f4dcf8942fe1cc6fb4e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_635d62ccba7f178d0d1030f27e02e8bb
    def get_inputs(self):
        return [
            paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            paddle.to_tensor([2047, 1], dtype='int32').reshape([2]),
        ]


class TestPrimitiveOp_cf71f70250fd361bd6032ca1973073f0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bb177be340c9c58e2e08e7b48f772261
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([1, 4116, 4], dtype='int64').reshape([3]),
        ]


class TestPrimitiveOp_f59a66f1aa32b2b408215bf7298ca6e1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bb177be340c9c58e2e08e7b48f772261
    def get_inputs(self):
        return [
            paddle.to_tensor([15.989999771118164], dtype='float32').reshape([1]),
            paddle.to_tensor([1, 4116, 4], dtype='int64').reshape([3]),
        ]


class TestPrimitiveOp_04e4f68ea7658b7d53135cca108042bb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bb177be340c9c58e2e08e7b48f772261
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([1, 3, 21, 21, 1], dtype='int64').reshape([5]),
        ]


class TestPrimitiveOp_ffd150431a7352f6e221fd920f358216(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bb177be340c9c58e2e08e7b48f772261
    def get_inputs(self):
        return [
            paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            paddle.to_tensor([1, 3, 21, 21, 1], dtype='int64').reshape([5]),
        ]


class TestPrimitiveOp_989876748dcaaf130c5172bd38167ee9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bb177be340c9c58e2e08e7b48f772261
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([1, 624], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_806e1d798a3937479e580216eabd019c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bb177be340c9c58e2e08e7b48f772261
    def get_inputs(self):
        return [
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            paddle.to_tensor([1, 624], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_918c4ec7bdff4886c55411f2add5c2a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bb177be340c9c58e2e08e7b48f772261
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([171, 60], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_808e2ed2b4798bce0330ee01fcf5ab8c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bb177be340c9c58e2e08e7b48f772261
    def get_inputs(self):
        return [
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            paddle.to_tensor([171, 60], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_a3ad31e35e9017999e9f4e051e6bb166(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bb177be340c9c58e2e08e7b48f772261
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([10, 336], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_dd570ce40f77e5ed76d053a23a6917c7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bb177be340c9c58e2e08e7b48f772261
    def get_inputs(self):
        return [
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            paddle.to_tensor([10, 336], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_3f96dd92f9ed471a70294e2a5e9886ec(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bb177be340c9c58e2e08e7b48f772261
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([1, 3, 42, 42, 1], dtype='int64').reshape([5]),
        ]


class TestPrimitiveOp_f997e2c3735aee96705b435a84bf6d94(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bb177be340c9c58e2e08e7b48f772261
    def get_inputs(self):
        return [
            paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            paddle.to_tensor([1, 3, 42, 42, 1], dtype='int64').reshape([5]),
        ]


class TestPrimitiveOp_8e009d04f0dfec7c52afb9dae2e8f029(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bb177be340c9c58e2e08e7b48f772261
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([1, 480], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_acc88dcf6f34ae00e19db1f905565d2a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bb177be340c9c58e2e08e7b48f772261
    def get_inputs(self):
        return [
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            paddle.to_tensor([1, 480], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_d60a1c7f4b5909f8877ed8e26e45630c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bb177be340c9c58e2e08e7b48f772261
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([145, 36], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_d7f2c97e61de6181f452a734df65ac22(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bb177be340c9c58e2e08e7b48f772261
    def get_inputs(self):
        return [
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            paddle.to_tensor([145, 36], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_d711ae08c1d3006027730ee9765758dd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bb177be340c9c58e2e08e7b48f772261
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([1, 3, 84, 84, 1], dtype='int64').reshape([5]),
        ]


class TestPrimitiveOp_efcac7b2208f81cfeb6faf92cbe9d9ab(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bb177be340c9c58e2e08e7b48f772261
    def get_inputs(self):
        return [
            paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            paddle.to_tensor([1, 3, 84, 84, 1], dtype='int64').reshape([5]),
        ]


class TestPrimitiveOp_a3ad31e35e9017999e9f4e051e6bb166(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bb177be340c9c58e2e08e7b48f772261
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([10, 336], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_dd570ce40f77e5ed76d053a23a6917c7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bb177be340c9c58e2e08e7b48f772261
    def get_inputs(self):
        return [
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            paddle.to_tensor([10, 336], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_4d606ac80205986c6d92933141f8c874(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ef22f064ba6b8bd11d8eacf9003d64b0
    def get_inputs(self):
        return [
            paddle.uniform([80, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([80, 80], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_0558c12c153eff97b684ba66c9829566(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ef22f064ba6b8bd11d8eacf9003d64b0
    def get_inputs(self):
        return [
            paddle.uniform([1, 80], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([80, 80], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_17de518e3b13bcea7a1a3e5ffa3badfc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ef22f064ba6b8bd11d8eacf9003d64b0
    def get_inputs(self):
        return [
            paddle.uniform([40, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([40, 40], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_7d43929735013f7d974cc6bc2ae434f5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ef22f064ba6b8bd11d8eacf9003d64b0
    def get_inputs(self):
        return [
            paddle.uniform([1, 40], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([40, 40], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_b195bf33fb7802c33b9eab94967a1f7b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ef22f064ba6b8bd11d8eacf9003d64b0
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.0], [32.0], [64.0], [96.0], [128.0], [160.0], [192.0], [224.0], [256.0], [288.0], [320.0], [352.0], [384.0], [416.0], [448.0], [480.0], [512.0], [544.0], [576.0], [608.0]], dtype='float32').reshape([20, 1]),
            paddle.to_tensor([20, 20], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_2c68e3b0ea838555e491185d88359aa0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ef22f064ba6b8bd11d8eacf9003d64b0
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.0, 32.0, 64.0, 96.0, 128.0, 160.0, 192.0, 224.0, 256.0, 288.0, 320.0, 352.0, 384.0, 416.0, 448.0, 480.0, 512.0, 544.0, 576.0, 608.0]], dtype='float32').reshape([1, 20]),
            paddle.to_tensor([20, 20], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_4d606ac80205986c6d92933141f8c874(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ef22f064ba6b8bd11d8eacf9003d64b0
    def get_inputs(self):
        return [
            paddle.uniform([80, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([80, 80], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_0558c12c153eff97b684ba66c9829566(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ef22f064ba6b8bd11d8eacf9003d64b0
    def get_inputs(self):
        return [
            paddle.uniform([1, 80], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([80, 80], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_17de518e3b13bcea7a1a3e5ffa3badfc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ef22f064ba6b8bd11d8eacf9003d64b0
    def get_inputs(self):
        return [
            paddle.uniform([40, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([40, 40], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_7d43929735013f7d974cc6bc2ae434f5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ef22f064ba6b8bd11d8eacf9003d64b0
    def get_inputs(self):
        return [
            paddle.uniform([1, 40], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([40, 40], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_eb8b6b5c417b43d438324d1c3690d630(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ef22f064ba6b8bd11d8eacf9003d64b0
    def get_inputs(self):
        return [
            paddle.to_tensor([[16.0], [48.0], [80.0], [112.0], [144.0], [176.0], [208.0], [240.0], [272.0], [304.0], [336.0], [368.0], [400.0], [432.0], [464.0], [496.0], [528.0], [560.0], [592.0], [624.0]], dtype='float32').reshape([20, 1]),
            paddle.to_tensor([20, 20], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_f35a14bfe5d3f8647cd490e370037e74(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ef22f064ba6b8bd11d8eacf9003d64b0
    def get_inputs(self):
        return [
            paddle.to_tensor([[16.0, 48.0, 80.0, 112.0, 144.0, 176.0, 208.0, 240.0, 272.0, 304.0, 336.0, 368.0, 400.0, 432.0, 464.0, 496.0, 528.0, 560.0, 592.0, 624.0]], dtype='float32').reshape([1, 20]),
            paddle.to_tensor([20, 20], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_2980e39836320e29169efaea004b7d9b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bb177be340c9c58e2e08e7b48f772261
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([5, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_4ef9d4807b5deb8bf6695c3367119de5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bb177be340c9c58e2e08e7b48f772261
    def get_inputs(self):
        return [
            paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            paddle.to_tensor([5, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_febeb43528fe6995f72e9c8e1899619d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_635d62ccba7f178d0d1030f27e02e8bb
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([1813, 1], dtype='int32').reshape([2]),
        ]


class TestPrimitiveOp_ad17f71a654a05ee42aefd0967d37e7f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_635d62ccba7f178d0d1030f27e02e8bb
    def get_inputs(self):
        return [
            paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            paddle.to_tensor([1813, 1], dtype='int32').reshape([2]),
        ]


class TestPrimitiveOp_febeb43528fe6995f72e9c8e1899619d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_635d62ccba7f178d0d1030f27e02e8bb
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([1813, 1], dtype='int32').reshape([2]),
        ]


class TestPrimitiveOp_ad17f71a654a05ee42aefd0967d37e7f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_635d62ccba7f178d0d1030f27e02e8bb
    def get_inputs(self):
        return [
            paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            paddle.to_tensor([1813, 1], dtype='int32').reshape([2]),
        ]


class TestPrimitiveOp_e4d3d066f04e26b6f48c5fe91a44d104(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bb177be340c9c58e2e08e7b48f772261
    def get_inputs(self):
        return [
            paddle.to_tensor([-2.0], dtype='float32').reshape([1]),
            paddle.to_tensor([1, 3549, 4], dtype='int64').reshape([3]),
        ]


class TestPrimitiveOp_18db55c550622edba4afb9cf3047e816(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bb177be340c9c58e2e08e7b48f772261
    def get_inputs(self):
        return [
            paddle.to_tensor([15.989999771118164], dtype='float32').reshape([1]),
            paddle.to_tensor([1, 3549, 4], dtype='int64').reshape([3]),
        ]


class TestPrimitiveOp_94c929991ec022c063814a9494150048(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bb177be340c9c58e2e08e7b48f772261
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([1, 72], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_94fc6400921b7e66229f3b2070662e6f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bb177be340c9c58e2e08e7b48f772261
    def get_inputs(self):
        return [
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            paddle.to_tensor([1, 72], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_ac39e3b584d955ba41091f117f4c5404(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ef22f064ba6b8bd11d8eacf9003d64b0
    def get_inputs(self):
        return [
            paddle.to_tensor([[16.0], [48.0], [80.0], [112.0], [144.0], [176.0], [208.0], [240.0], [272.0], [304.0], [336.0], [368.0], [400.0], [432.0]], dtype='float32').reshape([14, 1]),
            paddle.to_tensor([14, 14], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_f16d8c3e70a491b90ea6e587314c1c5b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ef22f064ba6b8bd11d8eacf9003d64b0
    def get_inputs(self):
        return [
            paddle.to_tensor([[16.0, 48.0, 80.0, 112.0, 144.0, 176.0, 208.0, 240.0, 272.0, 304.0, 336.0, 368.0, 400.0, 432.0]], dtype='float32').reshape([1, 14]),
            paddle.to_tensor([14, 14], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_6d71e7aa28c05f4b9cd718a471e311fd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ef22f064ba6b8bd11d8eacf9003d64b0
    def get_inputs(self):
        return [
            paddle.to_tensor([[8.0], [24.0], [40.0], [56.0], [72.0], [88.0], [104.0], [120.0], [136.0], [152.0], [168.0], [184.0], [200.0], [216.0], [232.0], [248.0], [264.0], [280.0], [296.0], [312.0], [328.0], [344.0], [360.0], [376.0], [392.0], [408.0], [424.0], [440.0]], dtype='float32').reshape([28, 1]),
            paddle.to_tensor([28, 28], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_a04f5030d9ab19caa7d7725eb141cd52(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ef22f064ba6b8bd11d8eacf9003d64b0
    def get_inputs(self):
        return [
            paddle.to_tensor([[8.0, 24.0, 40.0, 56.0, 72.0, 88.0, 104.0, 120.0, 136.0, 152.0, 168.0, 184.0, 200.0, 216.0, 232.0, 248.0, 264.0, 280.0, 296.0, 312.0, 328.0, 344.0, 360.0, 376.0, 392.0, 408.0, 424.0, 440.0]], dtype='float32').reshape([1, 28]),
            paddle.to_tensor([28, 28], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_b608fbe93705e2c2bc1007b801080132(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ef22f064ba6b8bd11d8eacf9003d64b0
    def get_inputs(self):
        return [
            paddle.uniform([56, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([56, 56], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_102078a84b84cbf1030936872921852c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ef22f064ba6b8bd11d8eacf9003d64b0
    def get_inputs(self):
        return [
            paddle.uniform([1, 56], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([56, 56], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_85e8660bbcfdef5f9622049809e670ea(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_635d62ccba7f178d0d1030f27e02e8bb
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([3061, 1], dtype='int32').reshape([2]),
        ]


class TestPrimitiveOp_bea70f483c039c84ddedba2b35305e78(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_635d62ccba7f178d0d1030f27e02e8bb
    def get_inputs(self):
        return [
            paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            paddle.to_tensor([3061, 1], dtype='int32').reshape([2]),
        ]


class TestPrimitiveOp_85e8660bbcfdef5f9622049809e670ea(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_635d62ccba7f178d0d1030f27e02e8bb
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([3061, 1], dtype='int32').reshape([2]),
        ]


class TestPrimitiveOp_bea70f483c039c84ddedba2b35305e78(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_635d62ccba7f178d0d1030f27e02e8bb
    def get_inputs(self):
        return [
            paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            paddle.to_tensor([3061, 1], dtype='int32').reshape([2]),
        ]


class TestPrimitiveOp_e02b4c91b89ad0121855b541f445ed49(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bb177be340c9c58e2e08e7b48f772261
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([1, 6069, 4], dtype='int64').reshape([3]),
        ]


class TestPrimitiveOp_8c4ddc594b79388de6fb3b2ce3248f25(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bb177be340c9c58e2e08e7b48f772261
    def get_inputs(self):
        return [
            paddle.to_tensor([15.989999771118164], dtype='float32').reshape([1]),
            paddle.to_tensor([1, 6069, 4], dtype='int64').reshape([3]),
        ]


class TestPrimitiveOp_df30c5930bf63b8d8505d217b661c33a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bb177be340c9c58e2e08e7b48f772261
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([1, 3, 48, 48, 1], dtype='int64').reshape([5]),
        ]


class TestPrimitiveOp_d4eb92809d6216fa5db95d1dbd07d8bc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bb177be340c9c58e2e08e7b48f772261
    def get_inputs(self):
        return [
            paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            paddle.to_tensor([1, 3, 48, 48, 1], dtype='int64').reshape([5]),
        ]


class TestPrimitiveOp_04ae10a80360e89e4ba79bf7539bfdc4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bb177be340c9c58e2e08e7b48f772261
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([171, 240], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_bd574cd60d3ce3184a507aab40beb8b7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bb177be340c9c58e2e08e7b48f772261
    def get_inputs(self):
        return [
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            paddle.to_tensor([171, 240], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_6ea94c45d3da54b022fa39835a482e17(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bb177be340c9c58e2e08e7b48f772261
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([10, 240], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_e4132c04523685182827e51587dead29(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bb177be340c9c58e2e08e7b48f772261
    def get_inputs(self):
        return [
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            paddle.to_tensor([10, 240], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_b914717c7e6f97726f9974287f8d95f0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bb177be340c9c58e2e08e7b48f772261
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([9, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_c235656055d9047dd150ca169cb9386f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bb177be340c9c58e2e08e7b48f772261
    def get_inputs(self):
        return [
            paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            paddle.to_tensor([9, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_0af6ef3e89db771c2bf03932db126729(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bb177be340c9c58e2e08e7b48f772261
    def get_inputs(self):
        return [
            paddle.to_tensor([-10000000000.0], dtype='float32').reshape([1]),
            paddle.to_tensor([6, 1, 1], dtype='int64').reshape([3]),
        ]


class TestPrimitiveOp_0294de3f996c1c86f328d93f231fd00e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bb177be340c9c58e2e08e7b48f772261
    def get_inputs(self):
        return [
            paddle.to_tensor([4.135169982910156], dtype='float32').reshape([1]),
            paddle.to_tensor([6, 1, 1], dtype='int64').reshape([3]),
        ]


class TestPrimitiveOp_5587022d1a2c33f2dd784969bd30fe21(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8467b6650b538e5114f1e56ed72de4de
    def get_inputs(self):
        return [
            paddle.uniform([1, 150, 768], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, -1, -1], dtype='int64').reshape([3]),
        ]


class TestPrimitiveOp_444c63bccf95fee576c84653e4d2e85c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_635d62ccba7f178d0d1030f27e02e8bb
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([2062, 1], dtype='int32').reshape([2]),
        ]


class TestPrimitiveOp_eb5346e48c7ca75773b5007399dfbc22(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_635d62ccba7f178d0d1030f27e02e8bb
    def get_inputs(self):
        return [
            paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            paddle.to_tensor([2062, 1], dtype='int32').reshape([2]),
        ]


class TestPrimitiveOp_444c63bccf95fee576c84653e4d2e85c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_635d62ccba7f178d0d1030f27e02e8bb
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([2062, 1], dtype='int32').reshape([2]),
        ]


class TestPrimitiveOp_eb5346e48c7ca75773b5007399dfbc22(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_635d62ccba7f178d0d1030f27e02e8bb
    def get_inputs(self):
        return [
            paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            paddle.to_tensor([2062, 1], dtype='int32').reshape([2]),
        ]


class TestPrimitiveOp_cf71f70250fd361bd6032ca1973073f0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bb177be340c9c58e2e08e7b48f772261
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([1, 4116, 4], dtype='int64').reshape([3]),
        ]


class TestPrimitiveOp_f59a66f1aa32b2b408215bf7298ca6e1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bb177be340c9c58e2e08e7b48f772261
    def get_inputs(self):
        return [
            paddle.to_tensor([15.989999771118164], dtype='float32').reshape([1]),
            paddle.to_tensor([1, 4116, 4], dtype='int64').reshape([3]),
        ]


class TestPrimitiveOp_3f96dd92f9ed471a70294e2a5e9886ec(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bb177be340c9c58e2e08e7b48f772261
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([1, 3, 42, 42, 1], dtype='int64').reshape([5]),
        ]


class TestPrimitiveOp_f997e2c3735aee96705b435a84bf6d94(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bb177be340c9c58e2e08e7b48f772261
    def get_inputs(self):
        return [
            paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            paddle.to_tensor([1, 3, 42, 42, 1], dtype='int64').reshape([5]),
        ]


class TestPrimitiveOp_71903354d6dcfe7920698a8cdb1448db(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bb177be340c9c58e2e08e7b48f772261
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([1, 3, 92, 92, 1], dtype='int64').reshape([5]),
        ]


class TestPrimitiveOp_041977fb3ef2cc9cecb4043fdc03457e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bb177be340c9c58e2e08e7b48f772261
    def get_inputs(self):
        return [
            paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            paddle.to_tensor([1, 3, 92, 92, 1], dtype='int64').reshape([5]),
        ]


class TestPrimitiveOp_b1c37afacc250fd76fb45ba23fd7c775(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bb177be340c9c58e2e08e7b48f772261
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([1, 3, 38, 38, 1], dtype='int64').reshape([5]),
        ]


class TestPrimitiveOp_0a9042df2d21c2b6a82803d57b6b3c3c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bb177be340c9c58e2e08e7b48f772261
    def get_inputs(self):
        return [
            paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            paddle.to_tensor([1, 3, 38, 38, 1], dtype='int64').reshape([5]),
        ]


class TestPrimitiveOp_7df9350d51d286f6f26853209c3ab8e0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bb177be340c9c58e2e08e7b48f772261
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([1, 3, 22, 22, 1], dtype='int64').reshape([5]),
        ]


class TestPrimitiveOp_a3bfc8ea69c525c32b5ceb89c123134b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bb177be340c9c58e2e08e7b48f772261
    def get_inputs(self):
        return [
            paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            paddle.to_tensor([1, 3, 22, 22, 1], dtype='int64').reshape([5]),
        ]


class TestPrimitiveOp_6e02ad34981b3024d6847c554cf94947(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bb177be340c9c58e2e08e7b48f772261
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([145, 480], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_f58f8acf9d740505b1920a1213dd085e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bb177be340c9c58e2e08e7b48f772261
    def get_inputs(self):
        return [
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            paddle.to_tensor([145, 480], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_4faf51a6a67f74fe5775a2e85bd27473(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bb177be340c9c58e2e08e7b48f772261
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([10, 36], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_1843e6f0e4040a15f6f20d5e0b4ba14a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bb177be340c9c58e2e08e7b48f772261
    def get_inputs(self):
        return [
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            paddle.to_tensor([10, 36], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_304f9200d640733302a8364a1f9ce094(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ef22f064ba6b8bd11d8eacf9003d64b0
    def get_inputs(self):
        return [
            paddle.uniform([32, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([32, 32], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_d30d3063bd5031cd5ebf1c5363843d77(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ef22f064ba6b8bd11d8eacf9003d64b0
    def get_inputs(self):
        return [
            paddle.uniform([1, 32], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([32, 32], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_c90fd4ac49d63ccb07863a811455b22b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ef22f064ba6b8bd11d8eacf9003d64b0
    def get_inputs(self):
        return [
            paddle.uniform([64, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([64, 64], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_fe5b9aace679aadf366a5c89f204925d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ef22f064ba6b8bd11d8eacf9003d64b0
    def get_inputs(self):
        return [
            paddle.uniform([1, 64], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([64, 64], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_365a997aa685598a822a4ecac47cab33(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ef22f064ba6b8bd11d8eacf9003d64b0
    def get_inputs(self):
        return [
            paddle.uniform([128, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([128, 128], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_561e48ff9ab50f4c74d784ac93074cc8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ef22f064ba6b8bd11d8eacf9003d64b0
    def get_inputs(self):
        return [
            paddle.uniform([1, 128], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([128, 128], dtype='int64').reshape([2]),
        ]


class PrimitiveOp_a189d001ce756614568b2d5798d0af84(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.expand(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_34975f895ce95491dd7b893860e2f2ed(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a189d001ce756614568b2d5798d0af84
    def get_inputs(self):
        return [
            paddle.to_tensor([[-1.0], [-0.9166666865348816], [-0.8333333134651184], [-0.75], [-0.6666666865348816], [-0.5833333134651184], [-0.5], [-0.4166666567325592], [-0.3333333432674408], [-0.25], [-0.1666666716337204], [-0.0833333358168602], [5.551115123125783e-17], [0.0833333358168602], [0.1666666716337204], [0.25], [0.3333333432674408], [0.4166666567325592], [0.5], [0.5833333134651184], [0.6666666865348816], [0.75], [0.8333333134651184], [0.9166666865348816], [1.0]], dtype='float32').reshape([25, 1]),
            paddle.to_tensor([25, 38], dtype='int32').reshape([2]),
        ]


class TestPrimitiveOp_a5dcb253d6aa166cd87e3442d2470023(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a189d001ce756614568b2d5798d0af84
    def get_inputs(self):
        return [
            paddle.uniform([1, 38], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([25, 38], dtype='int32').reshape([2]),
        ]


class PrimitiveOp_ae89a67d1c568020fe84d4383460a117(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.expand(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_e318f7d8b0345a79a92983c09e82fa57(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ae89a67d1c568020fe84d4383460a117
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 25, 38], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, 1, 25, 38], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_e318f7d8b0345a79a92983c09e82fa57(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ae89a67d1c568020fe84d4383460a117
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 25, 38], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, 1, 25, 38], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_faca2d7364f3ef4cd5dc03e19effae14(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ef22f064ba6b8bd11d8eacf9003d64b0
    def get_inputs(self):
        return [
            paddle.uniform([96, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([96, 96], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_5c70299ec0fb0e4c6695b4b2e3f4768f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ef22f064ba6b8bd11d8eacf9003d64b0
    def get_inputs(self):
        return [
            paddle.uniform([1, 96], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([96, 96], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_65592b1866457f948e06a203e1f9bb48(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ef22f064ba6b8bd11d8eacf9003d64b0
    def get_inputs(self):
        return [
            paddle.uniform([48, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([48, 48], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_c82decaa67c461638a5f84a4a5800517(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ef22f064ba6b8bd11d8eacf9003d64b0
    def get_inputs(self):
        return [
            paddle.uniform([1, 48], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([48, 48], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_3be373c4ed5e6739d07f4b2a39569644(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ef22f064ba6b8bd11d8eacf9003d64b0
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.0], [32.0], [64.0], [96.0], [128.0], [160.0], [192.0], [224.0], [256.0], [288.0], [320.0], [352.0], [384.0], [416.0], [448.0], [480.0], [512.0], [544.0], [576.0], [608.0], [640.0], [672.0], [704.0], [736.0]], dtype='float32').reshape([24, 1]),
            paddle.to_tensor([24, 24], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_6320528ddc499dbc7d3d698f856cf38e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ef22f064ba6b8bd11d8eacf9003d64b0
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.0, 32.0, 64.0, 96.0, 128.0, 160.0, 192.0, 224.0, 256.0, 288.0, 320.0, 352.0, 384.0, 416.0, 448.0, 480.0, 512.0, 544.0, 576.0, 608.0, 640.0, 672.0, 704.0, 736.0]], dtype='float32').reshape([1, 24]),
            paddle.to_tensor([24, 24], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_faca2d7364f3ef4cd5dc03e19effae14(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ef22f064ba6b8bd11d8eacf9003d64b0
    def get_inputs(self):
        return [
            paddle.uniform([96, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([96, 96], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_5c70299ec0fb0e4c6695b4b2e3f4768f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ef22f064ba6b8bd11d8eacf9003d64b0
    def get_inputs(self):
        return [
            paddle.uniform([1, 96], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([96, 96], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_65592b1866457f948e06a203e1f9bb48(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ef22f064ba6b8bd11d8eacf9003d64b0
    def get_inputs(self):
        return [
            paddle.uniform([48, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([48, 48], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_c82decaa67c461638a5f84a4a5800517(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ef22f064ba6b8bd11d8eacf9003d64b0
    def get_inputs(self):
        return [
            paddle.uniform([1, 48], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([48, 48], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_25a9988883427cfd7d882cbe903cf6a7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ef22f064ba6b8bd11d8eacf9003d64b0
    def get_inputs(self):
        return [
            paddle.to_tensor([[16.0], [48.0], [80.0], [112.0], [144.0], [176.0], [208.0], [240.0], [272.0], [304.0], [336.0], [368.0], [400.0], [432.0], [464.0], [496.0], [528.0], [560.0], [592.0], [624.0], [656.0], [688.0], [720.0], [752.0]], dtype='float32').reshape([24, 1]),
            paddle.to_tensor([24, 24], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_6ecfc071df9e2ff97d8ecffab4dee354(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ef22f064ba6b8bd11d8eacf9003d64b0
    def get_inputs(self):
        return [
            paddle.to_tensor([[16.0, 48.0, 80.0, 112.0, 144.0, 176.0, 208.0, 240.0, 272.0, 304.0, 336.0, 368.0, 400.0, 432.0, 464.0, 496.0, 528.0, 560.0, 592.0, 624.0, 656.0, 688.0, 720.0, 752.0]], dtype='float32').reshape([1, 24]),
            paddle.to_tensor([24, 24], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_30a4e95c2ce9862c421082b8e3c5461b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bb177be340c9c58e2e08e7b48f772261
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([1, 3, 44, 44, 1], dtype='int64').reshape([5]),
        ]


class TestPrimitiveOp_12cea2544800279a346c1bf2c607ac7a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bb177be340c9c58e2e08e7b48f772261
    def get_inputs(self):
        return [
            paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            paddle.to_tensor([1, 3, 44, 44, 1], dtype='int64').reshape([5]),
        ]


class TestPrimitiveOp_304f9200d640733302a8364a1f9ce094(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ef22f064ba6b8bd11d8eacf9003d64b0
    def get_inputs(self):
        return [
            paddle.uniform([32, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([32, 32], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_d30d3063bd5031cd5ebf1c5363843d77(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ef22f064ba6b8bd11d8eacf9003d64b0
    def get_inputs(self):
        return [
            paddle.uniform([1, 32], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([32, 32], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_c90fd4ac49d63ccb07863a811455b22b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ef22f064ba6b8bd11d8eacf9003d64b0
    def get_inputs(self):
        return [
            paddle.uniform([64, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([64, 64], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_fe5b9aace679aadf366a5c89f204925d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ef22f064ba6b8bd11d8eacf9003d64b0
    def get_inputs(self):
        return [
            paddle.uniform([1, 64], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([64, 64], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_365a997aa685598a822a4ecac47cab33(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ef22f064ba6b8bd11d8eacf9003d64b0
    def get_inputs(self):
        return [
            paddle.uniform([128, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([128, 128], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_561e48ff9ab50f4c74d784ac93074cc8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ef22f064ba6b8bd11d8eacf9003d64b0
    def get_inputs(self):
        return [
            paddle.uniform([1, 128], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([128, 128], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_9e25aec6b7b4dd11ffe094c010c5c7fa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bb177be340c9c58e2e08e7b48f772261
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([1, 3, 11, 11, 1], dtype='int64').reshape([5]),
        ]


class TestPrimitiveOp_dc4e3b92605222b4f8b7eb45cb406e66(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bb177be340c9c58e2e08e7b48f772261
    def get_inputs(self):
        return [
            paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            paddle.to_tensor([1, 3, 11, 11, 1], dtype='int64').reshape([5]),
        ]


class TestPrimitiveOp_51115d4b459503d0c084c1cf732970c0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bb177be340c9c58e2e08e7b48f772261
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_3fe9228ade3fa68304a8320b413c8ba6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bb177be340c9c58e2e08e7b48f772261
    def get_inputs(self):
        return [
            paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_c4d0f0e6eceade314688c31e456e7ef2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ef22f064ba6b8bd11d8eacf9003d64b0
    def get_inputs(self):
        return [
            paddle.uniform([68, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([68, 68], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_9ce642978b713a25bb56d8189dfdd541(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ef22f064ba6b8bd11d8eacf9003d64b0
    def get_inputs(self):
        return [
            paddle.uniform([1, 68], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([68, 68], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_7cdb3a09f342a2aecf14044ec13d56bb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ef22f064ba6b8bd11d8eacf9003d64b0
    def get_inputs(self):
        return [
            paddle.uniform([34, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([34, 34], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_4ba644a02e2cf77104b8fd1056b51950(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ef22f064ba6b8bd11d8eacf9003d64b0
    def get_inputs(self):
        return [
            paddle.uniform([1, 34], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([34, 34], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_bf2e90d0968a54d0ff1fee1d5e8cb285(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ef22f064ba6b8bd11d8eacf9003d64b0
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.0], [32.0], [64.0], [96.0], [128.0], [160.0], [192.0], [224.0], [256.0], [288.0], [320.0], [352.0], [384.0], [416.0], [448.0], [480.0], [512.0]], dtype='float32').reshape([17, 1]),
            paddle.to_tensor([17, 17], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_cd8dae9dffd8ddeb6c249177c2bd1611(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ef22f064ba6b8bd11d8eacf9003d64b0
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.0, 32.0, 64.0, 96.0, 128.0, 160.0, 192.0, 224.0, 256.0, 288.0, 320.0, 352.0, 384.0, 416.0, 448.0, 480.0, 512.0]], dtype='float32').reshape([1, 17]),
            paddle.to_tensor([17, 17], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_c4d0f0e6eceade314688c31e456e7ef2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ef22f064ba6b8bd11d8eacf9003d64b0
    def get_inputs(self):
        return [
            paddle.uniform([68, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([68, 68], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_9ce642978b713a25bb56d8189dfdd541(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ef22f064ba6b8bd11d8eacf9003d64b0
    def get_inputs(self):
        return [
            paddle.uniform([1, 68], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([68, 68], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_7cdb3a09f342a2aecf14044ec13d56bb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ef22f064ba6b8bd11d8eacf9003d64b0
    def get_inputs(self):
        return [
            paddle.uniform([34, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([34, 34], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_4ba644a02e2cf77104b8fd1056b51950(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ef22f064ba6b8bd11d8eacf9003d64b0
    def get_inputs(self):
        return [
            paddle.uniform([1, 34], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([34, 34], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_f2bbe35750fb52b1e192c59d9bee023f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ef22f064ba6b8bd11d8eacf9003d64b0
    def get_inputs(self):
        return [
            paddle.to_tensor([[16.0], [48.0], [80.0], [112.0], [144.0], [176.0], [208.0], [240.0], [272.0], [304.0], [336.0], [368.0], [400.0], [432.0], [464.0], [496.0], [528.0]], dtype='float32').reshape([17, 1]),
            paddle.to_tensor([17, 17], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_f4c817cf3261f49d8d6beee372e233cd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ef22f064ba6b8bd11d8eacf9003d64b0
    def get_inputs(self):
        return [
            paddle.to_tensor([[16.0, 48.0, 80.0, 112.0, 144.0, 176.0, 208.0, 240.0, 272.0, 304.0, 336.0, 368.0, 400.0, 432.0, 464.0, 496.0, 528.0]], dtype='float32').reshape([1, 17]),
            paddle.to_tensor([17, 17], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_9e25aec6b7b4dd11ffe094c010c5c7fa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bb177be340c9c58e2e08e7b48f772261
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([1, 3, 11, 11, 1], dtype='int64').reshape([5]),
        ]


class TestPrimitiveOp_dc4e3b92605222b4f8b7eb45cb406e66(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bb177be340c9c58e2e08e7b48f772261
    def get_inputs(self):
        return [
            paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            paddle.to_tensor([1, 3, 11, 11, 1], dtype='int64').reshape([5]),
        ]


class TestPrimitiveOp_04e4f68ea7658b7d53135cca108042bb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bb177be340c9c58e2e08e7b48f772261
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([1, 3, 21, 21, 1], dtype='int64').reshape([5]),
        ]


class TestPrimitiveOp_ffd150431a7352f6e221fd920f358216(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bb177be340c9c58e2e08e7b48f772261
    def get_inputs(self):
        return [
            paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            paddle.to_tensor([1, 3, 21, 21, 1], dtype='int64').reshape([5]),
        ]


class TestPrimitiveOp_70975347f19f4a6357563d870bee4439(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bb177be340c9c58e2e08e7b48f772261
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([1, 3, 23, 23, 1], dtype='int64').reshape([5]),
        ]


class TestPrimitiveOp_a49682bf2589be3a70802c9ca6d607be(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bb177be340c9c58e2e08e7b48f772261
    def get_inputs(self):
        return [
            paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            paddle.to_tensor([1, 3, 23, 23, 1], dtype='int64').reshape([5]),
        ]


class TestPrimitiveOp_1a97af2452ffc4b3ea5ef9faafecd71e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_635d62ccba7f178d0d1030f27e02e8bb
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([5526, 1], dtype='int32').reshape([2]),
        ]


class TestPrimitiveOp_cd53e717ea1492bf2adbee284cf5a59d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_635d62ccba7f178d0d1030f27e02e8bb
    def get_inputs(self):
        return [
            paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            paddle.to_tensor([5526, 1], dtype='int32').reshape([2]),
        ]


class TestPrimitiveOp_1a97af2452ffc4b3ea5ef9faafecd71e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_635d62ccba7f178d0d1030f27e02e8bb
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([5526, 1], dtype='int32').reshape([2]),
        ]


class TestPrimitiveOp_cd53e717ea1492bf2adbee284cf5a59d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_635d62ccba7f178d0d1030f27e02e8bb
    def get_inputs(self):
        return [
            paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            paddle.to_tensor([5526, 1], dtype='int32').reshape([2]),
        ]


class TestPrimitiveOp_59af6e7fc641e9f1b4092ac47af003f1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bb177be340c9c58e2e08e7b48f772261
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([1, 11109, 4], dtype='int64').reshape([3]),
        ]


class TestPrimitiveOp_5bd33a205fbbf3981e0c885aef88c8d0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bb177be340c9c58e2e08e7b48f772261
    def get_inputs(self):
        return [
            paddle.to_tensor([15.989999771118164], dtype='float32').reshape([1]),
            paddle.to_tensor([1, 11109, 4], dtype='int64').reshape([3]),
        ]


class TestPrimitiveOp_44eb28e6938baf3f769dec71b9e57b08(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bb177be340c9c58e2e08e7b48f772261
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([1, 3, 19, 19, 1], dtype='int64').reshape([5]),
        ]


class TestPrimitiveOp_faf3c4a9f0c942a4b2156c5dc76cab18(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bb177be340c9c58e2e08e7b48f772261
    def get_inputs(self):
        return [
            paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            paddle.to_tensor([1, 3, 19, 19, 1], dtype='int64').reshape([5]),
        ]


class TestPrimitiveOp_30a4e95c2ce9862c421082b8e3c5461b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bb177be340c9c58e2e08e7b48f772261
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([1, 3, 44, 44, 1], dtype='int64').reshape([5]),
        ]


class TestPrimitiveOp_12cea2544800279a346c1bf2c607ac7a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bb177be340c9c58e2e08e7b48f772261
    def get_inputs(self):
        return [
            paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            paddle.to_tensor([1, 3, 44, 44, 1], dtype='int64').reshape([5]),
        ]


class TestPrimitiveOp_ad2331b694f0b3f177a2e1bf29fa0eb5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_635d62ccba7f178d0d1030f27e02e8bb
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([1071, 1], dtype='int32').reshape([2]),
        ]


class TestPrimitiveOp_98a7d29d50e1b6872d612bc49dbe4954(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_635d62ccba7f178d0d1030f27e02e8bb
    def get_inputs(self):
        return [
            paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            paddle.to_tensor([1071, 1], dtype='int32').reshape([2]),
        ]


class TestPrimitiveOp_ad2331b694f0b3f177a2e1bf29fa0eb5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_635d62ccba7f178d0d1030f27e02e8bb
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([1071, 1], dtype='int32').reshape([2]),
        ]


class TestPrimitiveOp_98a7d29d50e1b6872d612bc49dbe4954(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_635d62ccba7f178d0d1030f27e02e8bb
    def get_inputs(self):
        return [
            paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            paddle.to_tensor([1071, 1], dtype='int32').reshape([2]),
        ]


class TestPrimitiveOp_aa92b4145a0eb29e20f0e88f73414d5b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bb177be340c9c58e2e08e7b48f772261
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([1, 2100, 4], dtype='int64').reshape([3]),
        ]


class TestPrimitiveOp_3986f0646361666466a3fdb14d8885f2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bb177be340c9c58e2e08e7b48f772261
    def get_inputs(self):
        return [
            paddle.to_tensor([15.989999771118164], dtype='float32').reshape([1]),
            paddle.to_tensor([1, 2100, 4], dtype='int64').reshape([3]),
        ]


class TestPrimitiveOp_c358751a74d84288bbbdf266072c153d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a189d001ce756614568b2d5798d0af84
    def get_inputs(self):
        return [
            paddle.to_tensor([[-1.0], [-0.8947368264198303], [-0.7894737124443054], [-0.6842105388641357], [-0.5789473652839661], [-0.4736842215061188], [-0.3684210479259491], [-0.2631579041481018], [-0.15789473056793213], [-0.05263157933950424], [0.05263157933950424], [0.15789473056793213], [0.2631579041481018], [0.3684210479259491], [0.4736842215061188], [0.5789473652839661], [0.6842105388641357], [0.7894737124443054], [0.8947368264198303], [1.0]], dtype='float32').reshape([20, 1]),
            paddle.to_tensor([20, 30], dtype='int32').reshape([2]),
        ]


class TestPrimitiveOp_d9cf67547fe46d66dc3dd43a37534d16(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a189d001ce756614568b2d5798d0af84
    def get_inputs(self):
        return [
            paddle.to_tensor([[-1.0, -0.931034505367279, -0.8620689511299133, -0.7931034564971924, -0.7241379022598267, -0.6551724076271057, -0.5862069129943848, -0.517241358757019, -0.4482758641242981, -0.37931033968925476, -0.3103448152542114, -0.24137930572032928, -0.17241379618644714, -0.1034482792019844, -0.03448275849223137, 0.03448275849223137, 0.1034482792019844, 0.17241379618644714, 0.24137930572032928, 0.3103448152542114, 0.37931033968925476, 0.4482758641242981, 0.517241358757019, 0.5862069129943848, 0.6551724076271057, 0.7241379022598267, 0.7931034564971924, 0.8620689511299133, 0.931034505367279, 1.0]], dtype='float32').reshape([1, 30]),
            paddle.to_tensor([20, 30], dtype='int32').reshape([2]),
        ]


class TestPrimitiveOp_fa35a9a95236fd644b62fd2c6d83480e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ae89a67d1c568020fe84d4383460a117
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 20, 30], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, 1, 20, 30], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_fa35a9a95236fd644b62fd2c6d83480e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ae89a67d1c568020fe84d4383460a117
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 20, 30], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, 1, 20, 30], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_7da3d5b359094e070ded436cc1b76d75(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8467b6650b538e5114f1e56ed72de4de
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 768], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, -1, -1], dtype='int64').reshape([3]),
        ]


class TestPrimitiveOp_304f9200d640733302a8364a1f9ce094(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ef22f064ba6b8bd11d8eacf9003d64b0
    def get_inputs(self):
        return [
            paddle.uniform([32, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([32, 32], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_d30d3063bd5031cd5ebf1c5363843d77(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ef22f064ba6b8bd11d8eacf9003d64b0
    def get_inputs(self):
        return [
            paddle.uniform([1, 32], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([32, 32], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_c90fd4ac49d63ccb07863a811455b22b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ef22f064ba6b8bd11d8eacf9003d64b0
    def get_inputs(self):
        return [
            paddle.uniform([64, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([64, 64], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_fe5b9aace679aadf366a5c89f204925d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ef22f064ba6b8bd11d8eacf9003d64b0
    def get_inputs(self):
        return [
            paddle.uniform([1, 64], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([64, 64], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_365a997aa685598a822a4ecac47cab33(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ef22f064ba6b8bd11d8eacf9003d64b0
    def get_inputs(self):
        return [
            paddle.uniform([128, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([128, 128], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_561e48ff9ab50f4c74d784ac93074cc8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ef22f064ba6b8bd11d8eacf9003d64b0
    def get_inputs(self):
        return [
            paddle.uniform([1, 128], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([128, 128], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_c6a0ad2a9f365e0b6ba96e58052f948a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_635d62ccba7f178d0d1030f27e02e8bb
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([1760, 1], dtype='int32').reshape([2]),
        ]


class TestPrimitiveOp_43f953de76918f99e9f5cd684ee74516(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_635d62ccba7f178d0d1030f27e02e8bb
    def get_inputs(self):
        return [
            paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            paddle.to_tensor([1760, 1], dtype='int32').reshape([2]),
        ]


class TestPrimitiveOp_c6a0ad2a9f365e0b6ba96e58052f948a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_635d62ccba7f178d0d1030f27e02e8bb
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([1760, 1], dtype='int32').reshape([2]),
        ]


class TestPrimitiveOp_43f953de76918f99e9f5cd684ee74516(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_635d62ccba7f178d0d1030f27e02e8bb
    def get_inputs(self):
        return [
            paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            paddle.to_tensor([1760, 1], dtype='int32').reshape([2]),
        ]


class TestPrimitiveOp_0d9335fb39333a1e502cef5e7d8fd523(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bb177be340c9c58e2e08e7b48f772261
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([1, 3549, 4], dtype='int64').reshape([3]),
        ]


class TestPrimitiveOp_18db55c550622edba4afb9cf3047e816(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bb177be340c9c58e2e08e7b48f772261
    def get_inputs(self):
        return [
            paddle.to_tensor([15.989999771118164], dtype='float32').reshape([1]),
            paddle.to_tensor([1, 3549, 4], dtype='int64').reshape([3]),
        ]


class TestPrimitiveOp_5d469bdc862f8880608985ddb98f1b24(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bb177be340c9c58e2e08e7b48f772261
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([1, 3, 12, 12, 1], dtype='int64').reshape([5]),
        ]


class TestPrimitiveOp_3528cf6bba64dca0d91fe74c3a3ec476(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bb177be340c9c58e2e08e7b48f772261
    def get_inputs(self):
        return [
            paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            paddle.to_tensor([1, 3, 12, 12, 1], dtype='int64').reshape([5]),
        ]


class TestPrimitiveOp_304f9200d640733302a8364a1f9ce094(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ef22f064ba6b8bd11d8eacf9003d64b0
    def get_inputs(self):
        return [
            paddle.uniform([32, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([32, 32], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_d30d3063bd5031cd5ebf1c5363843d77(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ef22f064ba6b8bd11d8eacf9003d64b0
    def get_inputs(self):
        return [
            paddle.uniform([1, 32], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([32, 32], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_c90fd4ac49d63ccb07863a811455b22b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ef22f064ba6b8bd11d8eacf9003d64b0
    def get_inputs(self):
        return [
            paddle.uniform([64, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([64, 64], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_fe5b9aace679aadf366a5c89f204925d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ef22f064ba6b8bd11d8eacf9003d64b0
    def get_inputs(self):
        return [
            paddle.uniform([1, 64], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([64, 64], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_365a997aa685598a822a4ecac47cab33(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ef22f064ba6b8bd11d8eacf9003d64b0
    def get_inputs(self):
        return [
            paddle.uniform([128, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([128, 128], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_561e48ff9ab50f4c74d784ac93074cc8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ef22f064ba6b8bd11d8eacf9003d64b0
    def get_inputs(self):
        return [
            paddle.uniform([1, 128], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([128, 128], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_365a997aa685598a822a4ecac47cab33(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ef22f064ba6b8bd11d8eacf9003d64b0
    def get_inputs(self):
        return [
            paddle.uniform([128, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([128, 128], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_561e48ff9ab50f4c74d784ac93074cc8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ef22f064ba6b8bd11d8eacf9003d64b0
    def get_inputs(self):
        return [
            paddle.uniform([1, 128], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([128, 128], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_c90fd4ac49d63ccb07863a811455b22b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ef22f064ba6b8bd11d8eacf9003d64b0
    def get_inputs(self):
        return [
            paddle.uniform([64, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([64, 64], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_fe5b9aace679aadf366a5c89f204925d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ef22f064ba6b8bd11d8eacf9003d64b0
    def get_inputs(self):
        return [
            paddle.uniform([1, 64], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([64, 64], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_304f9200d640733302a8364a1f9ce094(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ef22f064ba6b8bd11d8eacf9003d64b0
    def get_inputs(self):
        return [
            paddle.uniform([32, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([32, 32], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_d30d3063bd5031cd5ebf1c5363843d77(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ef22f064ba6b8bd11d8eacf9003d64b0
    def get_inputs(self):
        return [
            paddle.uniform([1, 32], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([32, 32], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_a0a4ff32211169c1b939b88ccd83db1c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ef22f064ba6b8bd11d8eacf9003d64b0
    def get_inputs(self):
        return [
            paddle.to_tensor([[32.0], [96.0], [160.0], [224.0], [288.0], [352.0], [416.0], [480.0], [544.0], [608.0], [672.0], [736.0], [800.0], [864.0], [928.0], [992.0]], dtype='float32').reshape([16, 1]),
            paddle.to_tensor([16, 16], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_d530106e5350eae00503c7cdbc124ffe(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ef22f064ba6b8bd11d8eacf9003d64b0
    def get_inputs(self):
        return [
            paddle.to_tensor([[32.0, 96.0, 160.0, 224.0, 288.0, 352.0, 416.0, 480.0, 544.0, 608.0, 672.0, 736.0, 800.0, 864.0, 928.0, 992.0]], dtype='float32').reshape([1, 16]),
            paddle.to_tensor([16, 16], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_7df4af76ad857813d8bbb897fe957bb7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ef22f064ba6b8bd11d8eacf9003d64b0
    def get_inputs(self):
        return [
            paddle.to_tensor([[64.0], [192.0], [320.0], [448.0], [576.0], [704.0], [832.0], [960.0]], dtype='float32').reshape([8, 1]),
            paddle.to_tensor([8, 8], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_a7056a55453d7a90fc00e3812aa4f11e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ef22f064ba6b8bd11d8eacf9003d64b0
    def get_inputs(self):
        return [
            paddle.to_tensor([[64.0, 192.0, 320.0, 448.0, 576.0, 704.0, 832.0, 960.0]], dtype='float32').reshape([1, 8]),
            paddle.to_tensor([8, 8], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_6e63e3f17aa0821925628734c1601091(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bb177be340c9c58e2e08e7b48f772261
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([1, 672], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_f8e614e5a662b959525f55d3c2c21b9e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bb177be340c9c58e2e08e7b48f772261
    def get_inputs(self):
        return [
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            paddle.to_tensor([1, 672], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_ad1afa736c44ac02cb4677fc477434f5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bb177be340c9c58e2e08e7b48f772261
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([1, 3, 46, 46, 1], dtype='int64').reshape([5]),
        ]


class TestPrimitiveOp_fb6e0a778cc9826de9372cdaabc63715(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bb177be340c9c58e2e08e7b48f772261
    def get_inputs(self):
        return [
            paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            paddle.to_tensor([1, 3, 46, 46, 1], dtype='int64').reshape([5]),
        ]


class TestPrimitiveOp_a1712f09cafca8824168589cf67b0726(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bb177be340c9c58e2e08e7b48f772261
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([1, 156], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_00c11e10cf64b50b171b91575b3eb929(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bb177be340c9c58e2e08e7b48f772261
    def get_inputs(self):
        return [
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            paddle.to_tensor([1, 156], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_8ac7157f025dc7f91424274e20b7f8c9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bb177be340c9c58e2e08e7b48f772261
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([145, 336], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_513330871590bae936e1ce0eadde0c63(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bb177be340c9c58e2e08e7b48f772261
    def get_inputs(self):
        return [
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            paddle.to_tensor([145, 336], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_9a9e45465255ce72d25482cf7270d3dc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a189d001ce756614568b2d5798d0af84
    def get_inputs(self):
        return [
            paddle.to_tensor([[-1.0], [-0.8571428656578064], [-0.7142857313156128], [-0.5714285969734192], [-0.4285714328289032], [-0.2857142984867096], [-0.1428571492433548], [5.551115123125783e-17], [0.1428571492433548], [0.2857142984867096], [0.4285714328289032], [0.5714285969734192], [0.7142857313156128], [0.8571428656578064], [1.0]], dtype='float32').reshape([15, 1]),
            paddle.to_tensor([15, 25], dtype='int32').reshape([2]),
        ]


class TestPrimitiveOp_ba0fd31ed361c7f2880e94e772c486c5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a189d001ce756614568b2d5798d0af84
    def get_inputs(self):
        return [
            paddle.to_tensor([[-1.0, -0.9166666865348816, -0.8333333134651184, -0.75, -0.6666666865348816, -0.5833333134651184, -0.5, -0.4166666567325592, -0.3333333432674408, -0.25, -0.1666666716337204, -0.0833333358168602, 5.551115123125783e-17, 0.0833333358168602, 0.1666666716337204, 0.25, 0.3333333432674408, 0.4166666567325592, 0.5, 0.5833333134651184, 0.6666666865348816, 0.75, 0.8333333134651184, 0.9166666865348816, 1.0]], dtype='float32').reshape([1, 25]),
            paddle.to_tensor([15, 25], dtype='int32').reshape([2]),
        ]


class TestPrimitiveOp_07ca5d33edef62db8b2beed941c7c741(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ae89a67d1c568020fe84d4383460a117
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 15, 25], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, 1, 15, 25], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_07ca5d33edef62db8b2beed941c7c741(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ae89a67d1c568020fe84d4383460a117
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 15, 25], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, 1, 15, 25], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_9683024a38d7e36ea193edd657310930(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bb177be340c9c58e2e08e7b48f772261
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([1, 3, 76, 76, 1], dtype='int64').reshape([5]),
        ]


class TestPrimitiveOp_4f9bb068dd49253749689edbc1234eff(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bb177be340c9c58e2e08e7b48f772261
    def get_inputs(self):
        return [
            paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            paddle.to_tensor([1, 3, 76, 76, 1], dtype='int64').reshape([5]),
        ]


class TestPrimitiveOp_9ee4b5c252d4d0218eda2e157458b877(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bb177be340c9c58e2e08e7b48f772261
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([22, 480], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_546e443226b582244e716846c4b25376(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bb177be340c9c58e2e08e7b48f772261
    def get_inputs(self):
        return [
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            paddle.to_tensor([22, 480], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_896ee59385e0a1d0c382b0dfbcafc0b0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bb177be340c9c58e2e08e7b48f772261
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([1, 872], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_93f5ac24c4c24f8f80dd179dee2bbe73(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bb177be340c9c58e2e08e7b48f772261
    def get_inputs(self):
        return [
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            paddle.to_tensor([1, 872], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_df30c5930bf63b8d8505d217b661c33a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bb177be340c9c58e2e08e7b48f772261
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([1, 3, 48, 48, 1], dtype='int64').reshape([5]),
        ]


class TestPrimitiveOp_d4eb92809d6216fa5db95d1dbd07d8bc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bb177be340c9c58e2e08e7b48f772261
    def get_inputs(self):
        return [
            paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            paddle.to_tensor([1, 3, 48, 48, 1], dtype='int64').reshape([5]),
        ]


class TestPrimitiveOp_ac9cce63a5b54e059788fd4341c8f672(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bb177be340c9c58e2e08e7b48f772261
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([1, 3, 24, 24, 1], dtype='int64').reshape([5]),
        ]


class TestPrimitiveOp_0efefb3dbdfbaec909645753af2263ce(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bb177be340c9c58e2e08e7b48f772261
    def get_inputs(self):
        return [
            paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            paddle.to_tensor([1, 3, 24, 24, 1], dtype='int64').reshape([5]),
        ]


class TestPrimitiveOp_896ee59385e0a1d0c382b0dfbcafc0b0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bb177be340c9c58e2e08e7b48f772261
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([1, 872], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_93f5ac24c4c24f8f80dd179dee2bbe73(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bb177be340c9c58e2e08e7b48f772261
    def get_inputs(self):
        return [
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            paddle.to_tensor([1, 872], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_4d887cb62ab68abe4291063c4f8302df(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_635d62ccba7f178d0d1030f27e02e8bb
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([4204, 1], dtype='int32').reshape([2]),
        ]


class TestPrimitiveOp_c9aa819a8a8ee7a2d5e794d0e0817556(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_635d62ccba7f178d0d1030f27e02e8bb
    def get_inputs(self):
        return [
            paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            paddle.to_tensor([4204, 1], dtype='int32').reshape([2]),
        ]


class TestPrimitiveOp_4d887cb62ab68abe4291063c4f8302df(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_635d62ccba7f178d0d1030f27e02e8bb
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([4204, 1], dtype='int32').reshape([2]),
        ]


class TestPrimitiveOp_c9aa819a8a8ee7a2d5e794d0e0817556(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_635d62ccba7f178d0d1030f27e02e8bb
    def get_inputs(self):
        return [
            paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            paddle.to_tensor([4204, 1], dtype='int32').reshape([2]),
        ]


class TestPrimitiveOp_416b5f258181bdbc5f1457747ce571d2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bb177be340c9c58e2e08e7b48f772261
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([1, 8400, 4], dtype='int64').reshape([3]),
        ]


class TestPrimitiveOp_cc1c3d329d1a9c43c6f221b75fa47a07(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bb177be340c9c58e2e08e7b48f772261
    def get_inputs(self):
        return [
            paddle.to_tensor([15.989999771118164], dtype='float32').reshape([1]),
            paddle.to_tensor([1, 8400, 4], dtype='int64').reshape([3]),
        ]


class TestPrimitiveOp_b1c37afacc250fd76fb45ba23fd7c775(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bb177be340c9c58e2e08e7b48f772261
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([1, 3, 38, 38, 1], dtype='int64').reshape([5]),
        ]


class TestPrimitiveOp_0a9042df2d21c2b6a82803d57b6b3c3c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bb177be340c9c58e2e08e7b48f772261
    def get_inputs(self):
        return [
            paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            paddle.to_tensor([1, 3, 38, 38, 1], dtype='int64').reshape([5]),
        ]


class TestPrimitiveOp_07afefaee3888bab2ffc656939dd68d8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bb177be340c9c58e2e08e7b48f772261
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([1, 92], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_0e96776fdb6429b63cf76ff8da73f0fa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bb177be340c9c58e2e08e7b48f772261
    def get_inputs(self):
        return [
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            paddle.to_tensor([1, 92], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_3aebdcb6bcee922fe9c89b62b55b8de4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ef22f064ba6b8bd11d8eacf9003d64b0
    def get_inputs(self):
        return [
            paddle.uniform([72, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([72, 72], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_ae83f9a5093148903596db4f33aa213e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ef22f064ba6b8bd11d8eacf9003d64b0
    def get_inputs(self):
        return [
            paddle.uniform([1, 72], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([72, 72], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_b0997d3a7b5d7da8f4811957ec9a42db(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ef22f064ba6b8bd11d8eacf9003d64b0
    def get_inputs(self):
        return [
            paddle.uniform([36, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([36, 36], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_e1a6e664d165b18b718ba5fda75914f6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ef22f064ba6b8bd11d8eacf9003d64b0
    def get_inputs(self):
        return [
            paddle.uniform([1, 36], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([36, 36], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_2350cef6a8d945deb76a02632196c0fe(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ef22f064ba6b8bd11d8eacf9003d64b0
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.0], [32.0], [64.0], [96.0], [128.0], [160.0], [192.0], [224.0], [256.0], [288.0], [320.0], [352.0], [384.0], [416.0], [448.0], [480.0], [512.0], [544.0]], dtype='float32').reshape([18, 1]),
            paddle.to_tensor([18, 18], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_836f70ba5bc8801cd8996f9eda2f53f1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ef22f064ba6b8bd11d8eacf9003d64b0
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.0, 32.0, 64.0, 96.0, 128.0, 160.0, 192.0, 224.0, 256.0, 288.0, 320.0, 352.0, 384.0, 416.0, 448.0, 480.0, 512.0, 544.0]], dtype='float32').reshape([1, 18]),
            paddle.to_tensor([18, 18], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_3aebdcb6bcee922fe9c89b62b55b8de4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ef22f064ba6b8bd11d8eacf9003d64b0
    def get_inputs(self):
        return [
            paddle.uniform([72, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([72, 72], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_ae83f9a5093148903596db4f33aa213e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ef22f064ba6b8bd11d8eacf9003d64b0
    def get_inputs(self):
        return [
            paddle.uniform([1, 72], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([72, 72], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_b0997d3a7b5d7da8f4811957ec9a42db(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ef22f064ba6b8bd11d8eacf9003d64b0
    def get_inputs(self):
        return [
            paddle.uniform([36, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([36, 36], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_e1a6e664d165b18b718ba5fda75914f6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ef22f064ba6b8bd11d8eacf9003d64b0
    def get_inputs(self):
        return [
            paddle.uniform([1, 36], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([36, 36], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_15e756db946f55033282522d7e0b9818(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ef22f064ba6b8bd11d8eacf9003d64b0
    def get_inputs(self):
        return [
            paddle.to_tensor([[16.0], [48.0], [80.0], [112.0], [144.0], [176.0], [208.0], [240.0], [272.0], [304.0], [336.0], [368.0], [400.0], [432.0], [464.0], [496.0], [528.0], [560.0]], dtype='float32').reshape([18, 1]),
            paddle.to_tensor([18, 18], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_4aaf06112d1f7aa51b9a83a2c1120096(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ef22f064ba6b8bd11d8eacf9003d64b0
    def get_inputs(self):
        return [
            paddle.to_tensor([[16.0, 48.0, 80.0, 112.0, 144.0, 176.0, 208.0, 240.0, 272.0, 304.0, 336.0, 368.0, 400.0, 432.0, 464.0, 496.0, 528.0, 560.0]], dtype='float32').reshape([1, 18]),
            paddle.to_tensor([18, 18], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_c90fd4ac49d63ccb07863a811455b22b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ef22f064ba6b8bd11d8eacf9003d64b0
    def get_inputs(self):
        return [
            paddle.uniform([64, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([64, 64], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_fe5b9aace679aadf366a5c89f204925d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ef22f064ba6b8bd11d8eacf9003d64b0
    def get_inputs(self):
        return [
            paddle.uniform([1, 64], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([64, 64], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_304f9200d640733302a8364a1f9ce094(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ef22f064ba6b8bd11d8eacf9003d64b0
    def get_inputs(self):
        return [
            paddle.uniform([32, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([32, 32], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_d30d3063bd5031cd5ebf1c5363843d77(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ef22f064ba6b8bd11d8eacf9003d64b0
    def get_inputs(self):
        return [
            paddle.uniform([1, 32], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([32, 32], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_f60d6ac249a8f5da9768d3277dc532b1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ef22f064ba6b8bd11d8eacf9003d64b0
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.0], [32.0], [64.0], [96.0], [128.0], [160.0], [192.0], [224.0], [256.0], [288.0], [320.0], [352.0], [384.0], [416.0], [448.0], [480.0]], dtype='float32').reshape([16, 1]),
            paddle.to_tensor([16, 16], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_dfec6078e7ada6191d2a6425a8593b0a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ef22f064ba6b8bd11d8eacf9003d64b0
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.0, 32.0, 64.0, 96.0, 128.0, 160.0, 192.0, 224.0, 256.0, 288.0, 320.0, 352.0, 384.0, 416.0, 448.0, 480.0]], dtype='float32').reshape([1, 16]),
            paddle.to_tensor([16, 16], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_c90fd4ac49d63ccb07863a811455b22b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ef22f064ba6b8bd11d8eacf9003d64b0
    def get_inputs(self):
        return [
            paddle.uniform([64, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([64, 64], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_fe5b9aace679aadf366a5c89f204925d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ef22f064ba6b8bd11d8eacf9003d64b0
    def get_inputs(self):
        return [
            paddle.uniform([1, 64], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([64, 64], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_304f9200d640733302a8364a1f9ce094(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ef22f064ba6b8bd11d8eacf9003d64b0
    def get_inputs(self):
        return [
            paddle.uniform([32, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([32, 32], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_d30d3063bd5031cd5ebf1c5363843d77(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ef22f064ba6b8bd11d8eacf9003d64b0
    def get_inputs(self):
        return [
            paddle.uniform([1, 32], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([32, 32], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_1dd03f6e23d89dd8d2fba02bea9d3dea(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ef22f064ba6b8bd11d8eacf9003d64b0
    def get_inputs(self):
        return [
            paddle.to_tensor([[16.0], [48.0], [80.0], [112.0], [144.0], [176.0], [208.0], [240.0], [272.0], [304.0], [336.0], [368.0], [400.0], [432.0], [464.0], [496.0]], dtype='float32').reshape([16, 1]),
            paddle.to_tensor([16, 16], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_8cf7e69543bf739f155c14b5059d642b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ef22f064ba6b8bd11d8eacf9003d64b0
    def get_inputs(self):
        return [
            paddle.to_tensor([[16.0, 48.0, 80.0, 112.0, 144.0, 176.0, 208.0, 240.0, 272.0, 304.0, 336.0, 368.0, 400.0, 432.0, 464.0, 496.0]], dtype='float32').reshape([1, 16]),
            paddle.to_tensor([16, 16], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_c11c1bf302244a7ab718907f1902e11b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a189d001ce756614568b2d5798d0af84
    def get_inputs(self):
        return [
            paddle.to_tensor([[-1.0], [-0.9130434989929199], [-0.8260869383811951], [-0.739130437374115], [-0.6521739363670349], [-0.5652173757553101], [-0.47826087474823], [-0.3913043439388275], [-0.30434781312942505], [-0.21739129722118378], [-0.1304347813129425], [-0.043478261679410934], [0.043478261679410934], [0.1304347813129425], [0.21739129722118378], [0.30434781312942505], [0.3913043439388275], [0.47826087474823], [0.5652173757553101], [0.6521739363670349], [0.739130437374115], [0.8260869383811951], [0.9130434989929199], [1.0]], dtype='float32').reshape([24, 1]),
            paddle.to_tensor([24, 36], dtype='int32').reshape([2]),
        ]


class TestPrimitiveOp_38b8157640b367c49706143571b8ece7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a189d001ce756614568b2d5798d0af84
    def get_inputs(self):
        return [
            paddle.uniform([1, 36], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([24, 36], dtype='int32').reshape([2]),
        ]


class TestPrimitiveOp_7021bca090e7406dc578879409fc61d5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ae89a67d1c568020fe84d4383460a117
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 24, 36], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, 1, 24, 36], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_7021bca090e7406dc578879409fc61d5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ae89a67d1c568020fe84d4383460a117
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 24, 36], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, 1, 24, 36], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_69f15e8a97ba2eb4461b2009b61a5972(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bb177be340c9c58e2e08e7b48f772261
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([22, 36], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_bfca0f33ce1de979b25102e059a55c4a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bb177be340c9c58e2e08e7b48f772261
    def get_inputs(self):
        return [
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            paddle.to_tensor([22, 36], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_d8cea89b52957651a0766f3c99d62ef5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_635d62ccba7f178d0d1030f27e02e8bb
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([4680, 1], dtype='int32').reshape([2]),
        ]


class TestPrimitiveOp_7760c314dc47a4af2dd43a15837e7b41(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_635d62ccba7f178d0d1030f27e02e8bb
    def get_inputs(self):
        return [
            paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            paddle.to_tensor([4680, 1], dtype='int32').reshape([2]),
        ]


class TestPrimitiveOp_d8cea89b52957651a0766f3c99d62ef5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_635d62ccba7f178d0d1030f27e02e8bb
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([4680, 1], dtype='int32').reshape([2]),
        ]


class TestPrimitiveOp_7760c314dc47a4af2dd43a15837e7b41(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_635d62ccba7f178d0d1030f27e02e8bb
    def get_inputs(self):
        return [
            paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            paddle.to_tensor([4680, 1], dtype='int32').reshape([2]),
        ]


class TestPrimitiveOp_7b74ebf8d8c0402203e453eae8c080c8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bb177be340c9c58e2e08e7b48f772261
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([1, 9261, 4], dtype='int64').reshape([3]),
        ]


class TestPrimitiveOp_2181597f1f0a6db7a3b3bcf930847de9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bb177be340c9c58e2e08e7b48f772261
    def get_inputs(self):
        return [
            paddle.to_tensor([15.989999771118164], dtype='float32').reshape([1]),
            paddle.to_tensor([1, 9261, 4], dtype='int64').reshape([3]),
        ]


class TestPrimitiveOp_3415ffcfce55d2d2d93d8d5f59ca4f79(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bb177be340c9c58e2e08e7b48f772261
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([171, 36], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_8218869e04352356932e8cd066fabdca(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bb177be340c9c58e2e08e7b48f772261
    def get_inputs(self):
        return [
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            paddle.to_tensor([171, 36], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_c557037fb09078c76332824055e30ea0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_635d62ccba7f178d0d1030f27e02e8bb
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([3778, 1], dtype='int32').reshape([2]),
        ]


class TestPrimitiveOp_f17b9e865b72792f7ec3b501be905c8c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_635d62ccba7f178d0d1030f27e02e8bb
    def get_inputs(self):
        return [
            paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            paddle.to_tensor([3778, 1], dtype='int32').reshape([2]),
        ]


class TestPrimitiveOp_c557037fb09078c76332824055e30ea0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_635d62ccba7f178d0d1030f27e02e8bb
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([3778, 1], dtype='int32').reshape([2]),
        ]


class TestPrimitiveOp_f17b9e865b72792f7ec3b501be905c8c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_635d62ccba7f178d0d1030f27e02e8bb
    def get_inputs(self):
        return [
            paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            paddle.to_tensor([3778, 1], dtype='int32').reshape([2]),
        ]


class TestPrimitiveOp_12121fc53153c015fb54914b93644893(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bb177be340c9c58e2e08e7b48f772261
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([1, 7581, 4], dtype='int64').reshape([3]),
        ]


class TestPrimitiveOp_414f6a353a07792f71e01478aad8eda4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bb177be340c9c58e2e08e7b48f772261
    def get_inputs(self):
        return [
            paddle.to_tensor([15.989999771118164], dtype='float32').reshape([1]),
            paddle.to_tensor([1, 7581, 4], dtype='int64').reshape([3]),
        ]


class TestPrimitiveOp_2e0b91606fe82d3f1134fc7ff63fe219(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bb177be340c9c58e2e08e7b48f772261
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([1, 1248], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_287b00e9e52d54a355abb2cf32cecb8e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bb177be340c9c58e2e08e7b48f772261
    def get_inputs(self):
        return [
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            paddle.to_tensor([1, 1248], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_3a734d9f7737c97a34e59fd7b20ad90d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ef22f064ba6b8bd11d8eacf9003d64b0
    def get_inputs(self):
        return [
            paddle.to_tensor([[8.0], [24.0], [40.0], [56.0], [72.0], [88.0], [104.0], [120.0], [136.0], [152.0], [168.0], [184.0], [200.0], [216.0], [232.0], [248.0], [264.0], [280.0], [296.0], [312.0], [328.0], [344.0], [360.0], [376.0]], dtype='float32').reshape([24, 1]),
            paddle.to_tensor([24, 24], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_ef2cdeb58d4a021728d7f821f7d7fe31(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ef22f064ba6b8bd11d8eacf9003d64b0
    def get_inputs(self):
        return [
            paddle.to_tensor([[8.0, 24.0, 40.0, 56.0, 72.0, 88.0, 104.0, 120.0, 136.0, 152.0, 168.0, 184.0, 200.0, 216.0, 232.0, 248.0, 264.0, 280.0, 296.0, 312.0, 328.0, 344.0, 360.0, 376.0]], dtype='float32').reshape([1, 24]),
            paddle.to_tensor([24, 24], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_30584a039e931718b806cd846e6ee81f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bb177be340c9c58e2e08e7b48f772261
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([1, 120], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_886f00bd8642fd1f7aaeb9c40cf519ed(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bb177be340c9c58e2e08e7b48f772261
    def get_inputs(self):
        return [
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            paddle.to_tensor([1, 120], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_7df9350d51d286f6f26853209c3ab8e0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bb177be340c9c58e2e08e7b48f772261
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([1, 3, 22, 22, 1], dtype='int64').reshape([5]),
        ]


class TestPrimitiveOp_a3bfc8ea69c525c32b5ceb89c123134b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bb177be340c9c58e2e08e7b48f772261
    def get_inputs(self):
        return [
            paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            paddle.to_tensor([1, 3, 22, 22, 1], dtype='int64').reshape([5]),
        ]


class TestPrimitiveOp_5da0ca461dfb41a634747fbd7125ccfd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bb177be340c9c58e2e08e7b48f772261
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([22, 336], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_50721740a542b08286df1babfeaebc7f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bb177be340c9c58e2e08e7b48f772261
    def get_inputs(self):
        return [
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            paddle.to_tensor([22, 336], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_1f5825332ff8840b14eaf90e08341ed8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bb177be340c9c58e2e08e7b48f772261
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([145, 240], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_e11dc17a5952e7a9f03d9f63140a77a1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bb177be340c9c58e2e08e7b48f772261
    def get_inputs(self):
        return [
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            paddle.to_tensor([145, 240], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_6209c16c0aac985b0eb62bdd145b923e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bb177be340c9c58e2e08e7b48f772261
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([171, 336], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_6a5b5d28c752811c7ab8be47fde51d00(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bb177be340c9c58e2e08e7b48f772261
    def get_inputs(self):
        return [
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            paddle.to_tensor([171, 336], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_1992722a2d54537a7b9af6241cbd799b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bb177be340c9c58e2e08e7b48f772261
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([10, 32, 100, 2], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_8f81f39772989775a288a0c184b414ee(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bb177be340c9c58e2e08e7b48f772261
    def get_inputs(self):
        return [
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            paddle.to_tensor([10, 32, 100, 2], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_6209c16c0aac985b0eb62bdd145b923e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bb177be340c9c58e2e08e7b48f772261
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([171, 336], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_6a5b5d28c752811c7ab8be47fde51d00(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bb177be340c9c58e2e08e7b48f772261
    def get_inputs(self):
        return [
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            paddle.to_tensor([171, 336], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_d711ae08c1d3006027730ee9765758dd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bb177be340c9c58e2e08e7b48f772261
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([1, 3, 84, 84, 1], dtype='int64').reshape([5]),
        ]


class TestPrimitiveOp_efcac7b2208f81cfeb6faf92cbe9d9ab(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bb177be340c9c58e2e08e7b48f772261
    def get_inputs(self):
        return [
            paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            paddle.to_tensor([1, 3, 84, 84, 1], dtype='int64').reshape([5]),
        ]


class TestPrimitiveOp_95df634d4b6e8146f851acc3b603cf19(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bb177be340c9c58e2e08e7b48f772261
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([10, 480], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_1869d0798c1194f78af4deb8f5adcbde(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bb177be340c9c58e2e08e7b48f772261
    def get_inputs(self):
        return [
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            paddle.to_tensor([10, 480], dtype='int64').reshape([2]),
        ]




if __name__ == '__main__':
    unittest.main()