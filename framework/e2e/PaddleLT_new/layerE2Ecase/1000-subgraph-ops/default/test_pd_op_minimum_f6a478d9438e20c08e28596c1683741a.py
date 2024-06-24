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


class PrimitiveOp_f99ba83f8f250754b59e109499c9f3a8(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.minimum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_f09405b2a3fb8d95c6993b4563e18e46(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f99ba83f8f250754b59e109499c9f3a8
    def get_inputs(self):
        return [
            paddle.uniform([1799, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1799, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f09405b2a3fb8d95c6993b4563e18e46(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f99ba83f8f250754b59e109499c9f3a8
    def get_inputs(self):
        return [
            paddle.uniform([1799, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1799, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f09405b2a3fb8d95c6993b4563e18e46(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f99ba83f8f250754b59e109499c9f3a8
    def get_inputs(self):
        return [
            paddle.uniform([1799, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1799, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f09405b2a3fb8d95c6993b4563e18e46(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f99ba83f8f250754b59e109499c9f3a8
    def get_inputs(self):
        return [
            paddle.uniform([1799, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1799, 1], dtype='float32', min=0, max=0.5),
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


class TestPrimitiveOp_c4a72619e661d5b3365fbcfa8a090dba(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e098a4a1a4457c33ab52971f06001bb6
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.3287186026573181], [0.18020182847976685], [0.08684854209423065], [0.2041427493095398], [0.05390194058418274], [0.2627392113208771], [0.33099016547203064], [0.15110880136489868], [0.10229241102933884]], dtype='float32').reshape([9, 1]),
            paddle.to_tensor([[0.24578014016151428], [0.06570521742105484], [0.45486047863960266], [0.12536022067070007], [0.46158698201179504], [0.43161243200302124], [0.4087907373905182], [0.27460622787475586], [0.4056161344051361]], dtype='float32').reshape([9, 1]),
        ]


class TestPrimitiveOp_dfa5278b1550c5e331ff8397cd0b3aff(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e098a4a1a4457c33ab52971f06001bb6
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.33190086483955383], [0.10759332776069641], [0.42710933089256287], [0.49739205837249756], [0.17430728673934937], [0.4577294886112213], [0.2397078424692154], [0.07001930475234985], [0.03557616472244263]], dtype='float32').reshape([9, 1]),
            paddle.to_tensor([[0.0795329138636589], [0.2186332643032074], [0.14820031821727753], [0.020885052159428596], [0.27499523758888245], [0.19093403220176697], [0.2295200675725937], [0.4841283857822418], [0.15295632183551788]], dtype='float32').reshape([9, 1]),
        ]


class TestPrimitiveOp_04418b40dfa71cc7baf1e096b7d4695a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e098a4a1a4457c33ab52971f06001bb6
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.331709086894989], [0.4315752387046814], [0.41960468888282776], [0.3054076135158539], [0.3308812975883484], [0.10524086654186249], [0.22144579887390137], [0.42640355229377747], [0.1348181813955307]], dtype='float32').reshape([9, 1]),
            paddle.to_tensor([[0.04048364609479904], [0.24893270432949066], [0.0918775126338005], [0.3730376958847046], [0.39346981048583984], [0.15195290744304657], [0.4746951162815094], [0.14278598129749298], [0.43999600410461426]], dtype='float32').reshape([9, 1]),
        ]


class TestPrimitiveOp_66b1d4ea3b7c6581ee101400678b39c5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e098a4a1a4457c33ab52971f06001bb6
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.20715078711509705], [0.11182703822851181], [0.1563730388879776], [0.45039767026901245], [0.01656464673578739], [0.03139396384358406], [0.338609904050827], [0.35356906056404114], [0.4564360976219177]], dtype='float32').reshape([9, 1]),
            paddle.to_tensor([[0.37394869327545166], [0.4309453070163727], [0.26047682762145996], [0.11842523515224457], [0.4338553249835968], [0.27303066849708557], [0.35420629382133484], [0.2772969901561737], [0.2206316888332367]], dtype='float32').reshape([9, 1]),
        ]


class TestPrimitiveOp_f221cf49f5aff3a449e955aadaea8c88(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f99ba83f8f250754b59e109499c9f3a8
    def get_inputs(self):
        return [
            paddle.uniform([5504, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([5504, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f221cf49f5aff3a449e955aadaea8c88(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f99ba83f8f250754b59e109499c9f3a8
    def get_inputs(self):
        return [
            paddle.uniform([5504, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([5504, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f221cf49f5aff3a449e955aadaea8c88(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f99ba83f8f250754b59e109499c9f3a8
    def get_inputs(self):
        return [
            paddle.uniform([5504, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([5504, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f221cf49f5aff3a449e955aadaea8c88(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f99ba83f8f250754b59e109499c9f3a8
    def get_inputs(self):
        return [
            paddle.uniform([5504, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([5504, 1], dtype='float32', min=0, max=0.5),
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


class TestPrimitiveOp_2da8b64a49bd8c38890a77abab13bebe(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a9d8ec59a9524c49ea98c653ec112e69
    def get_inputs(self):
        return [
            paddle.to_tensor([0.14632879197597504, 0.3513852059841156, 0.4086536467075348, 0.2538435161113739, 0.2506648898124695, 0.38373863697052], dtype='float32').reshape([6]),
            paddle.to_tensor([0.44832438230514526, 0.10885077714920044, 0.4067733883857727, 0.3167877793312073, 0.16312891244888306, 0.40700143575668335], dtype='float32').reshape([6]),
        ]


class TestPrimitiveOp_8fd7906604a8fd898493bf43a62ea207(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a9d8ec59a9524c49ea98c653ec112e69
    def get_inputs(self):
        return [
            paddle.to_tensor([0.4688446819782257, 0.11460484564304352, 0.44962406158447266, 0.3697131276130676, 0.27422139048576355, 0.3214428424835205], dtype='float32').reshape([6]),
            paddle.to_tensor([0.09651322662830353, 0.20735882222652435, 0.14432112872600555, 0.38802286982536316, 0.4720301926136017, 0.22569142282009125], dtype='float32').reshape([6]),
        ]


class TestPrimitiveOp_38890fb11c4632463fde019df95fbf87(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a9d8ec59a9524c49ea98c653ec112e69
    def get_inputs(self):
        return [
            paddle.to_tensor([0.12323445081710815, 0.3513852059841156, 0.4086536467075348, 0.1541879028081894, 0.0009443100425414741, 0.38373863697052], dtype='float32').reshape([6]),
            paddle.to_tensor([0.24217796325683594, 0.3652135729789734, 0.3323742747306824, 0.19348189234733582, 0.23804683983325958, 0.18084128201007843], dtype='float32').reshape([6]),
        ]


class TestPrimitiveOp_122b119f9beaa3a914dab1cd68933893(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a9d8ec59a9524c49ea98c653ec112e69
    def get_inputs(self):
        return [
            paddle.to_tensor([0.4688446819782257, 0.11460484564304352, 0.44962406158447266, 0.3697131276130676, 0.20614933967590332, 0.04639579355716705], dtype='float32').reshape([6]),
            paddle.to_tensor([0.3871614336967468, 0.34158778190612793, 0.2761411964893341, 0.46681585907936096, 0.3594697415828705, 0.42803919315338135], dtype='float32').reshape([6]),
        ]


class TestPrimitiveOp_1575fc131fe9233cbaaa6954cd47f9ad(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f99ba83f8f250754b59e109499c9f3a8
    def get_inputs(self):
        return [
            paddle.uniform([1811, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1811, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_1575fc131fe9233cbaaa6954cd47f9ad(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f99ba83f8f250754b59e109499c9f3a8
    def get_inputs(self):
        return [
            paddle.uniform([1811, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1811, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_1575fc131fe9233cbaaa6954cd47f9ad(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f99ba83f8f250754b59e109499c9f3a8
    def get_inputs(self):
        return [
            paddle.uniform([1811, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1811, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_1575fc131fe9233cbaaa6954cd47f9ad(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f99ba83f8f250754b59e109499c9f3a8
    def get_inputs(self):
        return [
            paddle.uniform([1811, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1811, 1], dtype='float32', min=0, max=0.5),
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


class TestPrimitiveOp_03758227932ce0502720227feb25d4ae(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f99ba83f8f250754b59e109499c9f3a8
    def get_inputs(self):
        return [
            paddle.uniform([1559, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1559, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_03758227932ce0502720227feb25d4ae(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f99ba83f8f250754b59e109499c9f3a8
    def get_inputs(self):
        return [
            paddle.uniform([1559, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1559, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_03758227932ce0502720227feb25d4ae(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f99ba83f8f250754b59e109499c9f3a8
    def get_inputs(self):
        return [
            paddle.uniform([1559, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1559, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_03758227932ce0502720227feb25d4ae(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f99ba83f8f250754b59e109499c9f3a8
    def get_inputs(self):
        return [
            paddle.uniform([1559, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1559, 1], dtype='float32', min=0, max=0.5),
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


class TestPrimitiveOp_8691f0be10ccd455eba01ab2931d8e64(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e098a4a1a4457c33ab52971f06001bb6
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.20194382965564728]], dtype='float32').reshape([1, 1]),
            paddle.to_tensor([[0.18423700332641602]], dtype='float32').reshape([1, 1]),
        ]


class TestPrimitiveOp_4f1d45fd298a53ce666182fa41919778(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e098a4a1a4457c33ab52971f06001bb6
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.2484549731016159]], dtype='float32').reshape([1, 1]),
            paddle.to_tensor([[0.2613685429096222]], dtype='float32').reshape([1, 1]),
        ]


class TestPrimitiveOp_5b927f68abd23fd18df6e5e8b3204d9d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e098a4a1a4457c33ab52971f06001bb6
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.31702283024787903]], dtype='float32').reshape([1, 1]),
            paddle.to_tensor([[0.08592501282691956]], dtype='float32').reshape([1, 1]),
        ]


class TestPrimitiveOp_a63c0e7fa0899f5fbff0bc56fe801df9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e098a4a1a4457c33ab52971f06001bb6
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.22766174376010895]], dtype='float32').reshape([1, 1]),
            paddle.to_tensor([[0.4279927611351013]], dtype='float32').reshape([1, 1]),
        ]


class TestPrimitiveOp_b4f95528409890928a765989e0badaf0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e098a4a1a4457c33ab52971f06001bb6
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.1737920194864273], [0.04220198094844818], [0.4123152494430542], [0.2897668480873108], [0.47753188014030457], [0.0939614474773407]], dtype='float32').reshape([6, 1]),
            paddle.to_tensor([[0.28047975897789], [0.05378994718194008], [0.059069257229566574], [0.08161171525716782], [0.2019866406917572], [0.4812576174736023]], dtype='float32').reshape([6, 1]),
        ]


class TestPrimitiveOp_cacc11839df4fda7fdba53d4b27ca97b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e098a4a1a4457c33ab52971f06001bb6
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.0316222719848156], [0.3478626012802124], [0.4299014210700989], [0.05371994525194168], [0.1470690220594406], [0.3986620306968689]], dtype='float32').reshape([6, 1]),
            paddle.to_tensor([[0.0392889678478241], [0.246535524725914], [0.4039975106716156], [0.49785470962524414], [0.2619699537754059], [0.23315942287445068]], dtype='float32').reshape([6, 1]),
        ]


class TestPrimitiveOp_7f936a3039623a678c7f5f40f2d9eae4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e098a4a1a4457c33ab52971f06001bb6
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.21024076640605927], [0.28957539796829224], [0.1927700936794281], [0.2966075837612152], [0.0682995542883873], [0.04952847585082054]], dtype='float32').reshape([6, 1]),
            paddle.to_tensor([[0.2295941263437271], [0.34228232502937317], [0.27764323353767395], [0.19415970146656036], [0.2267504632472992], [0.06496386975049973]], dtype='float32').reshape([6, 1]),
        ]


class TestPrimitiveOp_320e9fa840c00598181acd23e2fe6bd8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e098a4a1a4457c33ab52971f06001bb6
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.25873905420303345], [0.47146061062812805], [0.007305239327251911], [0.09287451207637787], [0.24141444265842438], [0.4163098633289337]], dtype='float32').reshape([6, 1]),
            paddle.to_tensor([[0.3855139911174774], [0.164891317486763], [0.30431702733039856], [0.289742648601532], [0.23264625668525696], [0.44625771045684814]], dtype='float32').reshape([6, 1]),
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


class TestPrimitiveOp_7b866e0a538de0aaf52cb868cdcf4d4d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f99ba83f8f250754b59e109499c9f3a8
    def get_inputs(self):
        return [
            paddle.uniform([2066, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2066, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_7b866e0a538de0aaf52cb868cdcf4d4d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f99ba83f8f250754b59e109499c9f3a8
    def get_inputs(self):
        return [
            paddle.uniform([2066, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2066, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_7b866e0a538de0aaf52cb868cdcf4d4d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f99ba83f8f250754b59e109499c9f3a8
    def get_inputs(self):
        return [
            paddle.uniform([2066, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2066, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_7b866e0a538de0aaf52cb868cdcf4d4d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f99ba83f8f250754b59e109499c9f3a8
    def get_inputs(self):
        return [
            paddle.uniform([2066, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2066, 1], dtype='float32', min=0, max=0.5),
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


class TestPrimitiveOp_f84eeb6d48d09f1607946de15d9314e5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f99ba83f8f250754b59e109499c9f3a8
    def get_inputs(self):
        return [
            paddle.uniform([4618, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4618, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f84eeb6d48d09f1607946de15d9314e5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f99ba83f8f250754b59e109499c9f3a8
    def get_inputs(self):
        return [
            paddle.uniform([4618, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4618, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f84eeb6d48d09f1607946de15d9314e5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f99ba83f8f250754b59e109499c9f3a8
    def get_inputs(self):
        return [
            paddle.uniform([4618, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4618, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f84eeb6d48d09f1607946de15d9314e5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f99ba83f8f250754b59e109499c9f3a8
    def get_inputs(self):
        return [
            paddle.uniform([4618, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4618, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c018d9ab165db2bfee3428f7895cd6f7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f99ba83f8f250754b59e109499c9f3a8
    def get_inputs(self):
        return [
            paddle.uniform([1058, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1058, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c018d9ab165db2bfee3428f7895cd6f7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f99ba83f8f250754b59e109499c9f3a8
    def get_inputs(self):
        return [
            paddle.uniform([1058, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1058, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c018d9ab165db2bfee3428f7895cd6f7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f99ba83f8f250754b59e109499c9f3a8
    def get_inputs(self):
        return [
            paddle.uniform([1058, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1058, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c018d9ab165db2bfee3428f7895cd6f7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f99ba83f8f250754b59e109499c9f3a8
    def get_inputs(self):
        return [
            paddle.uniform([1058, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1058, 1], dtype='float32', min=0, max=0.5),
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


class TestPrimitiveOp_73b00a0ed24ac366365fdf4e3309a1ee(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e098a4a1a4457c33ab52971f06001bb6
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.20464277267456055], [0.3423576056957245], [0.0038826088421046734], [0.21522128582000732], [0.0541519820690155]], dtype='float32').reshape([5, 1]),
            paddle.to_tensor([[0.016903966665267944], [0.2075682431459427], [0.20523563027381897], [0.4772803485393524], [0.2485220581293106]], dtype='float32').reshape([5, 1]),
        ]


class TestPrimitiveOp_775a50952cb60d1c0e91de72e7fc851e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e098a4a1a4457c33ab52971f06001bb6
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.45856836438179016], [0.38703131675720215], [0.40157803893089294], [0.4099683463573456], [0.07010837644338608]], dtype='float32').reshape([5, 1]),
            paddle.to_tensor([[0.22383494675159454], [0.43285518884658813], [0.22455859184265137], [0.28253594040870667], [0.3625085949897766]], dtype='float32').reshape([5, 1]),
        ]


class TestPrimitiveOp_2567a2b1fd14283d0b94bf0e8bc537eb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e098a4a1a4457c33ab52971f06001bb6
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.4358735978603363], [0.2401391565799713], [0.48159608244895935], [0.43455803394317627], [0.0802948921918869]], dtype='float32').reshape([5, 1]),
            paddle.to_tensor([[0.4782654941082001], [0.15857505798339844], [0.3338254690170288], [0.004150536842644215], [0.4149826467037201]], dtype='float32').reshape([5, 1]),
        ]


class TestPrimitiveOp_f181ff2dcf8e78c32b494d9ecfbe2100(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e098a4a1a4457c33ab52971f06001bb6
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.10876166075468063], [0.376865416765213], [0.34192609786987305], [0.4847533106803894], [0.291609525680542]], dtype='float32').reshape([5, 1]),
            paddle.to_tensor([[0.3702232539653778], [0.3718237280845642], [0.47763171792030334], [0.08214692771434784], [0.16273191571235657]], dtype='float32').reshape([5, 1]),
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


class TestPrimitiveOp_d348b138517f5b44a453b983772e2416(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f99ba83f8f250754b59e109499c9f3a8
    def get_inputs(self):
        return [
            paddle.uniform([2402, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2402, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d348b138517f5b44a453b983772e2416(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f99ba83f8f250754b59e109499c9f3a8
    def get_inputs(self):
        return [
            paddle.uniform([2402, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2402, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d348b138517f5b44a453b983772e2416(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f99ba83f8f250754b59e109499c9f3a8
    def get_inputs(self):
        return [
            paddle.uniform([2402, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2402, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d348b138517f5b44a453b983772e2416(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f99ba83f8f250754b59e109499c9f3a8
    def get_inputs(self):
        return [
            paddle.uniform([2402, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2402, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_99193d593c7d938e6abf9da088e69041(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f99ba83f8f250754b59e109499c9f3a8
    def get_inputs(self):
        return [
            paddle.uniform([2993, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2993, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_99193d593c7d938e6abf9da088e69041(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f99ba83f8f250754b59e109499c9f3a8
    def get_inputs(self):
        return [
            paddle.uniform([2993, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2993, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_99193d593c7d938e6abf9da088e69041(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f99ba83f8f250754b59e109499c9f3a8
    def get_inputs(self):
        return [
            paddle.uniform([2993, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2993, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_99193d593c7d938e6abf9da088e69041(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f99ba83f8f250754b59e109499c9f3a8
    def get_inputs(self):
        return [
            paddle.uniform([2993, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2993, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c1e20bf7c6d11a0d9857a26d163511f8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f99ba83f8f250754b59e109499c9f3a8
    def get_inputs(self):
        return [
            paddle.uniform([3787, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3787, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c1e20bf7c6d11a0d9857a26d163511f8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f99ba83f8f250754b59e109499c9f3a8
    def get_inputs(self):
        return [
            paddle.uniform([3787, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3787, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c1e20bf7c6d11a0d9857a26d163511f8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f99ba83f8f250754b59e109499c9f3a8
    def get_inputs(self):
        return [
            paddle.uniform([3787, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3787, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c1e20bf7c6d11a0d9857a26d163511f8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f99ba83f8f250754b59e109499c9f3a8
    def get_inputs(self):
        return [
            paddle.uniform([3787, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3787, 1], dtype='float32', min=0, max=0.5),
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


class TestPrimitiveOp_a34bd05c51782f6c06d7254291bde3c1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e098a4a1a4457c33ab52971f06001bb6
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.33608508110046387], [0.36661604046821594], [0.08292681723833084], [0.2736373543739319]], dtype='float32').reshape([4, 1]),
            paddle.to_tensor([[0.38861891627311707], [0.17682288587093353], [0.2886216640472412], [0.04139469563961029]], dtype='float32').reshape([4, 1]),
        ]


class TestPrimitiveOp_a8bf026a2fd2ffb6c680a32d0915049a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e098a4a1a4457c33ab52971f06001bb6
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.2787860631942749], [0.025012454017996788], [0.35968491435050964], [0.26496902108192444]], dtype='float32').reshape([4, 1]),
            paddle.to_tensor([[0.4298492968082428], [0.2742873728275299], [0.4489744007587433], [0.11074841767549515]], dtype='float32').reshape([4, 1]),
        ]


class TestPrimitiveOp_d9bc708e0dcdde255e0525c9171133c8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e098a4a1a4457c33ab52971f06001bb6
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.12751974165439606], [0.4973313510417938], [0.4870723485946655], [0.49783867597579956]], dtype='float32').reshape([4, 1]),
            paddle.to_tensor([[0.16749481856822968], [0.22387433052062988], [0.30820056796073914], [0.2222341001033783]], dtype='float32').reshape([4, 1]),
        ]


class TestPrimitiveOp_cdaf4d0315531714988d2b92d8cddc88(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e098a4a1a4457c33ab52971f06001bb6
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.46310076117515564], [0.3204939365386963], [0.1550406515598297], [0.08015791326761246]], dtype='float32').reshape([4, 1]),
            paddle.to_tensor([[0.4111035466194153], [0.13314859569072723], [0.12051653861999512], [0.28060683608055115]], dtype='float32').reshape([4, 1]),
        ]


class TestPrimitiveOp_597cf94e419ab972a6f5764337744f86(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f99ba83f8f250754b59e109499c9f3a8
    def get_inputs(self):
        return [
            paddle.uniform([2114, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2114, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_597cf94e419ab972a6f5764337744f86(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f99ba83f8f250754b59e109499c9f3a8
    def get_inputs(self):
        return [
            paddle.uniform([2114, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2114, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_597cf94e419ab972a6f5764337744f86(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f99ba83f8f250754b59e109499c9f3a8
    def get_inputs(self):
        return [
            paddle.uniform([2114, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2114, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_597cf94e419ab972a6f5764337744f86(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f99ba83f8f250754b59e109499c9f3a8
    def get_inputs(self):
        return [
            paddle.uniform([2114, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2114, 1], dtype='float32', min=0, max=0.5),
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


class TestPrimitiveOp_d02f7ad460676e82a14b85b6afc9deb6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f99ba83f8f250754b59e109499c9f3a8
    def get_inputs(self):
        return [
            paddle.uniform([4156, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4156, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d02f7ad460676e82a14b85b6afc9deb6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f99ba83f8f250754b59e109499c9f3a8
    def get_inputs(self):
        return [
            paddle.uniform([4156, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4156, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d02f7ad460676e82a14b85b6afc9deb6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f99ba83f8f250754b59e109499c9f3a8
    def get_inputs(self):
        return [
            paddle.uniform([4156, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4156, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d02f7ad460676e82a14b85b6afc9deb6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f99ba83f8f250754b59e109499c9f3a8
    def get_inputs(self):
        return [
            paddle.uniform([4156, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4156, 1], dtype='float32', min=0, max=0.5),
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