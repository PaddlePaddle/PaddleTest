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



class PrimitiveOp_0066edc5450243aed008b5625d34c300(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.minimum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None, None, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_10cb16380a7afdf4d331e664af1e9877(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0066edc5450243aed008b5625d34c300
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_10cb16380a7afdf4d331e664af1e9877(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0066edc5450243aed008b5625d34c300
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e5b7b003139e13a34964dcb32bd57754(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0066edc5450243aed008b5625d34c300
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e5b7b003139e13a34964dcb32bd57754(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0066edc5450243aed008b5625d34c300
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d2c7910b2cb0284536640cdb8fc09d75(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0066edc5450243aed008b5625d34c300
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d2c7910b2cb0284536640cdb8fc09d75(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0066edc5450243aed008b5625d34c300
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4ff5ac1f01dbb68c2a20bf579d1397fd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0066edc5450243aed008b5625d34c300
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4ff5ac1f01dbb68c2a20bf579d1397fd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0066edc5450243aed008b5625d34c300
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b523c2641be6e3a734ff3a89f7c127f0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0066edc5450243aed008b5625d34c300
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b523c2641be6e3a734ff3a89f7c127f0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0066edc5450243aed008b5625d34c300
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_dc85b5fb7b44ca2ebe66b7ecf956ba10(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0066edc5450243aed008b5625d34c300
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_dc85b5fb7b44ca2ebe66b7ecf956ba10(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0066edc5450243aed008b5625d34c300
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_10cb16380a7afdf4d331e664af1e9877(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0066edc5450243aed008b5625d34c300
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_10cb16380a7afdf4d331e664af1e9877(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0066edc5450243aed008b5625d34c300
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_905ed21d1f906d3c4fd030d26dc7cddc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0066edc5450243aed008b5625d34c300
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_905ed21d1f906d3c4fd030d26dc7cddc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0066edc5450243aed008b5625d34c300
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_8fdd71a8ebcf89d8c60536f4c9c25694(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0066edc5450243aed008b5625d34c300
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_8fdd71a8ebcf89d8c60536f4c9c25694(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0066edc5450243aed008b5625d34c300
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_e098a4a1a4457c33ab52971f06001bb6(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.minimum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_72711fcaad0ff7228ea4f98909b667cf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e098a4a1a4457c33ab52971f06001bb6
    def get_inputs(self):
        return [
            paddle.uniform([1787, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1787, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_72711fcaad0ff7228ea4f98909b667cf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e098a4a1a4457c33ab52971f06001bb6
    def get_inputs(self):
        return [
            paddle.uniform([1787, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1787, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_72711fcaad0ff7228ea4f98909b667cf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e098a4a1a4457c33ab52971f06001bb6
    def get_inputs(self):
        return [
            paddle.uniform([1787, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1787, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_72711fcaad0ff7228ea4f98909b667cf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e098a4a1a4457c33ab52971f06001bb6
    def get_inputs(self):
        return [
            paddle.uniform([1787, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1787, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_8149f267687e9b1bf6ad1803bc73028b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0066edc5450243aed008b5625d34c300
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_8149f267687e9b1bf6ad1803bc73028b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0066edc5450243aed008b5625d34c300
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6a04d52d9490f7cdde29a27a8978c858(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0066edc5450243aed008b5625d34c300
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6a04d52d9490f7cdde29a27a8978c858(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0066edc5450243aed008b5625d34c300
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c30b0a4686cc47360bcc22ffa08c2af1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0066edc5450243aed008b5625d34c300
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c30b0a4686cc47360bcc22ffa08c2af1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0066edc5450243aed008b5625d34c300
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b451c0a3602761548de06690b6b7fb73(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0066edc5450243aed008b5625d34c300
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b451c0a3602761548de06690b6b7fb73(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0066edc5450243aed008b5625d34c300
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_318057e41bdb81ed23b96b7c4d160ea2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e098a4a1a4457c33ab52971f06001bb6
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.4150841236114502], [0.17719584703445435], [0.21496637165546417], [0.07059020549058914], [0.36353129148483276], [0.49319419264793396], [0.1559973657131195], [0.15078814327716827], [0.12845014035701752]], dtype='float32').reshape([9, 1]),
            paddle.to_tensor([[0.4822169244289398], [0.2370811104774475], [0.3954120874404907], [0.3224271237850189], [0.47750574350357056], [0.06053764000535011], [0.37847888469696045], [0.3879892826080322], [0.2486654371023178]], dtype='float32').reshape([9, 1]),
        ]


class TestPrimitiveOp_61162bba7c4f6e79bf5fdf7f1e3617e6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e098a4a1a4457c33ab52971f06001bb6
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.16026167571544647], [0.3790070712566376], [0.16010837256908417], [0.2957713007926941], [0.01934489607810974], [0.0937335267663002], [0.43187767267227173], [0.11734545975923538], [0.11581700295209885]], dtype='float32').reshape([9, 1]),
            paddle.to_tensor([[0.34030860662460327], [0.10591290891170502], [0.057278454303741455], [0.4635138511657715], [0.3712193965911865], [0.026201101019978523], [0.08061874657869339], [0.24713134765625], [0.4871062934398651]], dtype='float32').reshape([9, 1]),
        ]


class TestPrimitiveOp_84cffc936978d9c2d8ad8270ffed5c22(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e098a4a1a4457c33ab52971f06001bb6
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.1411658227443695], [0.12203264236450195], [0.10209637880325317], [0.05470713600516319], [0.4240017235279083], [0.45557984709739685], [0.3452005386352539], [0.2678227722644806], [0.22818942368030548]], dtype='float32').reshape([9, 1]),
            paddle.to_tensor([[0.323263943195343], [0.04598228260874748], [0.21097292006015778], [0.17394885420799255], [0.20643334090709686], [0.37540435791015625], [0.2720523476600647], [0.16043679416179657], [0.37276360392570496]], dtype='float32').reshape([9, 1]),
        ]


class TestPrimitiveOp_8acf34dd7ca52b91d020e69ebc7718ac(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e098a4a1a4457c33ab52971f06001bb6
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.045104995369911194], [0.327657550573349], [0.39709436893463135], [0.4026397466659546], [0.3111007809638977], [0.0928461104631424], [0.22239001095294952], [0.3817371726036072], [0.28099653124809265]], dtype='float32').reshape([9, 1]),
            paddle.to_tensor([[0.1299373209476471], [0.19547255337238312], [0.46365925669670105], [0.21294736862182617], [0.36426010727882385], [0.28336191177368164], [0.45397958159446716], [0.21277481317520142], [0.38122108578681946]], dtype='float32').reshape([9, 1]),
        ]


class TestPrimitiveOp_c870b9a99430e7d03d8c00137e4af812(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e098a4a1a4457c33ab52971f06001bb6
    def get_inputs(self):
        return [
            paddle.uniform([5524, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([5524, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c870b9a99430e7d03d8c00137e4af812(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e098a4a1a4457c33ab52971f06001bb6
    def get_inputs(self):
        return [
            paddle.uniform([5524, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([5524, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c870b9a99430e7d03d8c00137e4af812(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e098a4a1a4457c33ab52971f06001bb6
    def get_inputs(self):
        return [
            paddle.uniform([5524, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([5524, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c870b9a99430e7d03d8c00137e4af812(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e098a4a1a4457c33ab52971f06001bb6
    def get_inputs(self):
        return [
            paddle.uniform([5524, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([5524, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_a9d8ec59a9524c49ea98c653ec112e69(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.minimum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None], dtype='float32'),
            paddle.static.InputSpec(shape=[None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_37dc841a7dd0c573e43275f713d4e672(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a9d8ec59a9524c49ea98c653ec112e69
    def get_inputs(self):
        return [
            paddle.to_tensor([0.11504115909337997, 0.29690277576446533, 0.4288880527019501, 0.2790885865688324, 0.33935993909835815, 0.18947434425354004], dtype='float32').reshape([6]),
            paddle.to_tensor([0.07288321852684021, 0.4954397678375244, 0.14402969181537628, 0.24536927044391632, 0.27300581336021423, 0.4197378158569336], dtype='float32').reshape([6]),
        ]


class TestPrimitiveOp_861661b02fa88c029384fe376ad42f04(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a9d8ec59a9524c49ea98c653ec112e69
    def get_inputs(self):
        return [
            paddle.to_tensor([0.44253310561180115, 0.23894467949867249, 0.467602014541626, 0.40499556064605713, 0.43618178367614746, 0.3588932752609253], dtype='float32').reshape([6]),
            paddle.to_tensor([0.47645989060401917, 0.17707553505897522, 0.3205617666244507, 0.45086607336997986, 0.13450734317302704, 0.038778964430093765], dtype='float32').reshape([6]),
        ]


class TestPrimitiveOp_ab5acd2b2246775602606fd59250657f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a9d8ec59a9524c49ea98c653ec112e69
    def get_inputs(self):
        return [
            paddle.to_tensor([0.11504115909337997, 0.1618838608264923, 0.13696600496768951, 0.2790885865688324, 0.33935993909835815, 0.18947434425354004], dtype='float32').reshape([6]),
            paddle.to_tensor([0.4411706030368805, 0.49970942735671997, 0.4972732365131378, 0.038729194551706314, 0.4908983111381531, 0.06910925358533859], dtype='float32').reshape([6]),
        ]


class TestPrimitiveOp_3b38f626d735684d3e66bc38773caf15(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a9d8ec59a9524c49ea98c653ec112e69
    def get_inputs(self):
        return [
            paddle.to_tensor([0.44253310561180115, 0.23894467949867249, 0.467602014541626, 0.2677920460700989, 0.14295744895935059, 0.3588932752609253], dtype='float32').reshape([6]),
            paddle.to_tensor([0.22670212388038635, 0.28265973925590515, 0.26170220971107483, 0.45209547877311707, 0.20575770735740662, 0.2268163412809372], dtype='float32').reshape([6]),
        ]


class TestPrimitiveOp_3c37da13bc256c74becff632d5e461a3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e098a4a1a4457c33ab52971f06001bb6
    def get_inputs(self):
        return [
            paddle.uniform([1722, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1722, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3c37da13bc256c74becff632d5e461a3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e098a4a1a4457c33ab52971f06001bb6
    def get_inputs(self):
        return [
            paddle.uniform([1722, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1722, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3c37da13bc256c74becff632d5e461a3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e098a4a1a4457c33ab52971f06001bb6
    def get_inputs(self):
        return [
            paddle.uniform([1722, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1722, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3c37da13bc256c74becff632d5e461a3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e098a4a1a4457c33ab52971f06001bb6
    def get_inputs(self):
        return [
            paddle.uniform([1722, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1722, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4ff5ac1f01dbb68c2a20bf579d1397fd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0066edc5450243aed008b5625d34c300
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4ff5ac1f01dbb68c2a20bf579d1397fd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0066edc5450243aed008b5625d34c300
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6a04d52d9490f7cdde29a27a8978c858(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0066edc5450243aed008b5625d34c300
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6a04d52d9490f7cdde29a27a8978c858(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0066edc5450243aed008b5625d34c300
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_dc038ac9dc67704614749c9a0a57e706(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e098a4a1a4457c33ab52971f06001bb6
    def get_inputs(self):
        return [
            paddle.uniform([1565, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1565, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_dc038ac9dc67704614749c9a0a57e706(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e098a4a1a4457c33ab52971f06001bb6
    def get_inputs(self):
        return [
            paddle.uniform([1565, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1565, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_dc038ac9dc67704614749c9a0a57e706(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e098a4a1a4457c33ab52971f06001bb6
    def get_inputs(self):
        return [
            paddle.uniform([1565, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1565, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_dc038ac9dc67704614749c9a0a57e706(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e098a4a1a4457c33ab52971f06001bb6
    def get_inputs(self):
        return [
            paddle.uniform([1565, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1565, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_dc85b5fb7b44ca2ebe66b7ecf956ba10(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0066edc5450243aed008b5625d34c300
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_dc85b5fb7b44ca2ebe66b7ecf956ba10(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0066edc5450243aed008b5625d34c300
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e5b7b003139e13a34964dcb32bd57754(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0066edc5450243aed008b5625d34c300
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e5b7b003139e13a34964dcb32bd57754(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0066edc5450243aed008b5625d34c300
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_56f89efb93f464c16847721d61660db9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e098a4a1a4457c33ab52971f06001bb6
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.3264276087284088]], dtype='float32').reshape([1, 1]),
            paddle.to_tensor([[0.036440495401620865]], dtype='float32').reshape([1, 1]),
        ]


class TestPrimitiveOp_44e824969e0cca0fb3729cb997edd586(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e098a4a1a4457c33ab52971f06001bb6
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.06712997704744339]], dtype='float32').reshape([1, 1]),
            paddle.to_tensor([[0.4700833559036255]], dtype='float32').reshape([1, 1]),
        ]


class TestPrimitiveOp_f3a21f3460541d207c10b5b65f4d07a6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e098a4a1a4457c33ab52971f06001bb6
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.2607162296772003]], dtype='float32').reshape([1, 1]),
            paddle.to_tensor([[0.018885329365730286]], dtype='float32').reshape([1, 1]),
        ]


class TestPrimitiveOp_84c45cc3a9c170a032533b41c2db4dd1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e098a4a1a4457c33ab52971f06001bb6
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.4917317032814026]], dtype='float32').reshape([1, 1]),
            paddle.to_tensor([[0.052278950810432434]], dtype='float32').reshape([1, 1]),
        ]


class TestPrimitiveOp_1edea9c75dafea675880c834211076b8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e098a4a1a4457c33ab52971f06001bb6
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.18295827507972717], [0.3151487112045288], [0.04074244946241379], [0.2054625153541565], [0.1159159243106842], [0.48689642548561096]], dtype='float32').reshape([6, 1]),
            paddle.to_tensor([[0.010900466702878475], [0.010215289890766144], [0.06574226170778275], [0.16454607248306274], [0.4259689450263977], [0.14867933094501495]], dtype='float32').reshape([6, 1]),
        ]


class TestPrimitiveOp_b99ba91e51126ba11ddcc593df48a477(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e098a4a1a4457c33ab52971f06001bb6
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.3233656883239746], [0.2852637469768524], [0.37613576650619507], [0.07651174813508987], [0.14534544944763184], [0.2512975037097931]], dtype='float32').reshape([6, 1]),
            paddle.to_tensor([[0.12707766890525818], [0.28534623980522156], [0.09402547776699066], [0.03946004435420036], [0.36656326055526733], [0.2210235297679901]], dtype='float32').reshape([6, 1]),
        ]


class TestPrimitiveOp_4b3d61ecf199e12430d6bfe9339c065e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e098a4a1a4457c33ab52971f06001bb6
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.4822092354297638], [0.006724483333528042], [0.2837485671043396], [0.4681859612464905], [0.3815319240093231], [0.058043427765369415]], dtype='float32').reshape([6, 1]),
            paddle.to_tensor([[0.2975405156612396], [0.48823976516723633], [0.1359703093767166], [0.4426943063735962], [0.2658000886440277], [0.3857698142528534]], dtype='float32').reshape([6, 1]),
        ]


class TestPrimitiveOp_0a202cdba048992ca51ec4404e2b63dc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e098a4a1a4457c33ab52971f06001bb6
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.4434575140476227], [0.09507346153259277], [0.22235330939292908], [0.23332111537456512], [0.21799568831920624], [0.2868433892726898]], dtype='float32').reshape([6, 1]),
            paddle.to_tensor([[0.03964489325881004], [0.19159133732318878], [0.018977906554937363], [0.11470313370227814], [0.4990856349468231], [0.38143986463546753]], dtype='float32').reshape([6, 1]),
        ]


class TestPrimitiveOp_8149f267687e9b1bf6ad1803bc73028b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0066edc5450243aed008b5625d34c300
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_8149f267687e9b1bf6ad1803bc73028b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0066edc5450243aed008b5625d34c300
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d2c7910b2cb0284536640cdb8fc09d75(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0066edc5450243aed008b5625d34c300
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d2c7910b2cb0284536640cdb8fc09d75(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0066edc5450243aed008b5625d34c300
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d74fa50d8032f7c84b16ee1fe0664445(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e098a4a1a4457c33ab52971f06001bb6
    def get_inputs(self):
        return [
            paddle.uniform([2034, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2034, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d74fa50d8032f7c84b16ee1fe0664445(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e098a4a1a4457c33ab52971f06001bb6
    def get_inputs(self):
        return [
            paddle.uniform([2034, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2034, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d74fa50d8032f7c84b16ee1fe0664445(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e098a4a1a4457c33ab52971f06001bb6
    def get_inputs(self):
        return [
            paddle.uniform([2034, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2034, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d74fa50d8032f7c84b16ee1fe0664445(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e098a4a1a4457c33ab52971f06001bb6
    def get_inputs(self):
        return [
            paddle.uniform([2034, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2034, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c8e8f5ae9bc50034fdd6f600553a7fc7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0066edc5450243aed008b5625d34c300
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c8e8f5ae9bc50034fdd6f600553a7fc7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0066edc5450243aed008b5625d34c300
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_89ad13924cb104ea59361e5449c25c1b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e098a4a1a4457c33ab52971f06001bb6
    def get_inputs(self):
        return [
            paddle.uniform([4667, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4667, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_89ad13924cb104ea59361e5449c25c1b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e098a4a1a4457c33ab52971f06001bb6
    def get_inputs(self):
        return [
            paddle.uniform([4667, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4667, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_89ad13924cb104ea59361e5449c25c1b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e098a4a1a4457c33ab52971f06001bb6
    def get_inputs(self):
        return [
            paddle.uniform([4667, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4667, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_89ad13924cb104ea59361e5449c25c1b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e098a4a1a4457c33ab52971f06001bb6
    def get_inputs(self):
        return [
            paddle.uniform([4667, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4667, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_7b7c8129827b225486dd765e877c651b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e098a4a1a4457c33ab52971f06001bb6
    def get_inputs(self):
        return [
            paddle.uniform([1052, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1052, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_7b7c8129827b225486dd765e877c651b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e098a4a1a4457c33ab52971f06001bb6
    def get_inputs(self):
        return [
            paddle.uniform([1052, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1052, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_7b7c8129827b225486dd765e877c651b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e098a4a1a4457c33ab52971f06001bb6
    def get_inputs(self):
        return [
            paddle.uniform([1052, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1052, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_7b7c8129827b225486dd765e877c651b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e098a4a1a4457c33ab52971f06001bb6
    def get_inputs(self):
        return [
            paddle.uniform([1052, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1052, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b523c2641be6e3a734ff3a89f7c127f0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0066edc5450243aed008b5625d34c300
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b523c2641be6e3a734ff3a89f7c127f0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0066edc5450243aed008b5625d34c300
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c30b0a4686cc47360bcc22ffa08c2af1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0066edc5450243aed008b5625d34c300
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c30b0a4686cc47360bcc22ffa08c2af1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0066edc5450243aed008b5625d34c300
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9f4971474d3a085a8aed0ffcba8004e6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e098a4a1a4457c33ab52971f06001bb6
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.1842043548822403], [0.4558430314064026], [0.181528702378273], [0.03595537319779396], [0.40070411562919617]], dtype='float32').reshape([5, 1]),
            paddle.to_tensor([[0.24109651148319244], [0.29673632979393005], [0.1784258335828781], [0.23225410282611847], [0.32011303305625916]], dtype='float32').reshape([5, 1]),
        ]


class TestPrimitiveOp_3d7810eb16b4e0173eade689c4fa68c7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e098a4a1a4457c33ab52971f06001bb6
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.4554417133331299], [0.41594210267066956], [0.3925545811653137], [0.020104756578803062], [0.04694260284304619]], dtype='float32').reshape([5, 1]),
            paddle.to_tensor([[0.33795109391212463], [0.16441994905471802], [0.22406062483787537], [0.06259459257125854], [0.2541954815387726]], dtype='float32').reshape([5, 1]),
        ]


class TestPrimitiveOp_2692bb532b3512ac9f105fe336f8388c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e098a4a1a4457c33ab52971f06001bb6
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.49922287464141846], [0.22875554859638214], [0.4021463096141815], [0.29854851961135864], [0.11341562122106552]], dtype='float32').reshape([5, 1]),
            paddle.to_tensor([[0.16909852623939514], [0.42314374446868896], [0.04594412073493004], [0.446805477142334], [0.03256781026721001]], dtype='float32').reshape([5, 1]),
        ]


class TestPrimitiveOp_fad612fd95cbbaacea3d7db0904671c7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e098a4a1a4457c33ab52971f06001bb6
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.2773558795452118], [0.38949447870254517], [0.37457266449928284], [0.32656681537628174], [0.42194125056266785]], dtype='float32').reshape([5, 1]),
            paddle.to_tensor([[0.36494573950767517], [0.43061792850494385], [0.0018265079706907272], [0.0859675481915474], [0.4702182114124298]], dtype='float32').reshape([5, 1]),
        ]


class TestPrimitiveOp_8fdd71a8ebcf89d8c60536f4c9c25694(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0066edc5450243aed008b5625d34c300
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_8fdd71a8ebcf89d8c60536f4c9c25694(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0066edc5450243aed008b5625d34c300
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b451c0a3602761548de06690b6b7fb73(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0066edc5450243aed008b5625d34c300
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b451c0a3602761548de06690b6b7fb73(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0066edc5450243aed008b5625d34c300
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_fcec9f6bda8fecd4654627c6235699d8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0066edc5450243aed008b5625d34c300
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_fcec9f6bda8fecd4654627c6235699d8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0066edc5450243aed008b5625d34c300
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_526e9bb67a1788fd9fd7be85526b612e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e098a4a1a4457c33ab52971f06001bb6
    def get_inputs(self):
        return [
            paddle.uniform([2378, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2378, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_526e9bb67a1788fd9fd7be85526b612e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e098a4a1a4457c33ab52971f06001bb6
    def get_inputs(self):
        return [
            paddle.uniform([2378, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2378, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_526e9bb67a1788fd9fd7be85526b612e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e098a4a1a4457c33ab52971f06001bb6
    def get_inputs(self):
        return [
            paddle.uniform([2378, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2378, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_526e9bb67a1788fd9fd7be85526b612e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e098a4a1a4457c33ab52971f06001bb6
    def get_inputs(self):
        return [
            paddle.uniform([2378, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2378, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d700796e042b6c9952a82c3f74b379fb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e098a4a1a4457c33ab52971f06001bb6
    def get_inputs(self):
        return [
            paddle.uniform([3105, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3105, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d700796e042b6c9952a82c3f74b379fb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e098a4a1a4457c33ab52971f06001bb6
    def get_inputs(self):
        return [
            paddle.uniform([3105, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3105, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d700796e042b6c9952a82c3f74b379fb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e098a4a1a4457c33ab52971f06001bb6
    def get_inputs(self):
        return [
            paddle.uniform([3105, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3105, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d700796e042b6c9952a82c3f74b379fb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e098a4a1a4457c33ab52971f06001bb6
    def get_inputs(self):
        return [
            paddle.uniform([3105, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3105, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f9b50df914d2dd7ab18fc04bb98a9922(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e098a4a1a4457c33ab52971f06001bb6
    def get_inputs(self):
        return [
            paddle.uniform([3832, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3832, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f9b50df914d2dd7ab18fc04bb98a9922(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e098a4a1a4457c33ab52971f06001bb6
    def get_inputs(self):
        return [
            paddle.uniform([3832, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3832, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f9b50df914d2dd7ab18fc04bb98a9922(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e098a4a1a4457c33ab52971f06001bb6
    def get_inputs(self):
        return [
            paddle.uniform([3832, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3832, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f9b50df914d2dd7ab18fc04bb98a9922(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e098a4a1a4457c33ab52971f06001bb6
    def get_inputs(self):
        return [
            paddle.uniform([3832, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3832, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c8e8f5ae9bc50034fdd6f600553a7fc7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0066edc5450243aed008b5625d34c300
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c8e8f5ae9bc50034fdd6f600553a7fc7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0066edc5450243aed008b5625d34c300
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_035f362a08ae64bcbce6536562b4997b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e098a4a1a4457c33ab52971f06001bb6
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.22413021326065063], [0.35453227162361145], [0.04643124341964722], [0.005095055792480707]], dtype='float32').reshape([4, 1]),
            paddle.to_tensor([[0.08569023758172989], [0.19257299602031708], [0.43132296204566956], [0.04351089894771576]], dtype='float32').reshape([4, 1]),
        ]


class TestPrimitiveOp_ff2e4719ac6c56041d275a53e35da682(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e098a4a1a4457c33ab52971f06001bb6
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.47936275601387024], [0.34922024607658386], [0.17143593728542328], [0.052620213478803635]], dtype='float32').reshape([4, 1]),
            paddle.to_tensor([[0.451505184173584], [0.06611757725477219], [0.2026786357164383], [0.04553063213825226]], dtype='float32').reshape([4, 1]),
        ]


class TestPrimitiveOp_9b90495f9b370912b4700cdb2b469aae(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e098a4a1a4457c33ab52971f06001bb6
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.17769232392311096], [0.3656418025493622], [0.4181656539440155], [0.2764999568462372]], dtype='float32').reshape([4, 1]),
            paddle.to_tensor([[0.3369443416595459], [0.4299197494983673], [0.4530823826789856], [0.46458011865615845]], dtype='float32').reshape([4, 1]),
        ]


class TestPrimitiveOp_9e9250bc6d47c436ee224a8699051891(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e098a4a1a4457c33ab52971f06001bb6
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.15277644991874695], [0.1881343126296997], [0.42179977893829346], [0.2802838683128357]], dtype='float32').reshape([4, 1]),
            paddle.to_tensor([[0.3807269036769867], [0.04921408370137215], [0.1457361876964569], [0.25481411814689636]], dtype='float32').reshape([4, 1]),
        ]


class TestPrimitiveOp_9180646fbae06133df2756ba050e55cf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e098a4a1a4457c33ab52971f06001bb6
    def get_inputs(self):
        return [
            paddle.uniform([2087, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2087, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9180646fbae06133df2756ba050e55cf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e098a4a1a4457c33ab52971f06001bb6
    def get_inputs(self):
        return [
            paddle.uniform([2087, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2087, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9180646fbae06133df2756ba050e55cf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e098a4a1a4457c33ab52971f06001bb6
    def get_inputs(self):
        return [
            paddle.uniform([2087, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2087, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9180646fbae06133df2756ba050e55cf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e098a4a1a4457c33ab52971f06001bb6
    def get_inputs(self):
        return [
            paddle.uniform([2087, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2087, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_fcec9f6bda8fecd4654627c6235699d8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0066edc5450243aed008b5625d34c300
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_fcec9f6bda8fecd4654627c6235699d8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0066edc5450243aed008b5625d34c300
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_905ed21d1f906d3c4fd030d26dc7cddc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0066edc5450243aed008b5625d34c300
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_905ed21d1f906d3c4fd030d26dc7cddc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0066edc5450243aed008b5625d34c300
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_0e88bc0cd04d415e65a266a1a8b5b37c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0066edc5450243aed008b5625d34c300
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_0e88bc0cd04d415e65a266a1a8b5b37c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0066edc5450243aed008b5625d34c300
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_39a66331983ce6bcaca1b0616807609f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e098a4a1a4457c33ab52971f06001bb6
    def get_inputs(self):
        return [
            paddle.uniform([4271, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4271, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_39a66331983ce6bcaca1b0616807609f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e098a4a1a4457c33ab52971f06001bb6
    def get_inputs(self):
        return [
            paddle.uniform([4271, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4271, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_39a66331983ce6bcaca1b0616807609f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e098a4a1a4457c33ab52971f06001bb6
    def get_inputs(self):
        return [
            paddle.uniform([4271, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4271, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_39a66331983ce6bcaca1b0616807609f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e098a4a1a4457c33ab52971f06001bb6
    def get_inputs(self):
        return [
            paddle.uniform([4271, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4271, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_0e88bc0cd04d415e65a266a1a8b5b37c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0066edc5450243aed008b5625d34c300
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_0e88bc0cd04d415e65a266a1a8b5b37c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0066edc5450243aed008b5625d34c300
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
        ]




if __name__ == '__main__':
    unittest.main()