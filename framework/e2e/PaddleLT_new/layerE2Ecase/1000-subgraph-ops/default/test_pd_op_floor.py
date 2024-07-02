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



class PrimitiveOp_a1461f5dac00b621fb40b0f8d89877da(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.floor(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 1, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_2bdb93a2d38c8f985acbe9e67a28f9a1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a1461f5dac00b621fb40b0f8d89877da
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[0.9953716397285461]]], [[[1.6395931243896484]]], [[[1.747664451599121]]], [[[1.8384348154067993]]], [[[1.5743753910064697]]], [[[1.288377046585083]]], [[[1.6506237983703613]]], [[[1.4782893657684326]]], [[[1.2311307191848755]]], [[[1.0404328107833862]]], [[[1.2389861345291138]]]], dtype='float32').reshape([11, 1, 1, 1]),
        ]


class TestPrimitiveOp_2add8be0f7b7ef5b92b6ae5b2887495d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a1461f5dac00b621fb40b0f8d89877da
    def get_inputs(self):
        return [
            paddle.uniform([43, 1, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_2add8be0f7b7ef5b92b6ae5b2887495d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a1461f5dac00b621fb40b0f8d89877da
    def get_inputs(self):
        return [
            paddle.uniform([43, 1, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_b47207d47005ed2d331b2c4c07e75725(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.floor(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 4], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_97016c6d8267fc4eece1c04241407a8e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b47207d47005ed2d331b2c4c07e75725
    def get_inputs(self):
        return [
            paddle.uniform([1756, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c4c051612f2f8b18048a6263ef3ba379(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a1461f5dac00b621fb40b0f8d89877da
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[1.5833258628845215]]], [[[1.8246928453445435]]], [[[1.8213376998901367]]], [[[1.8430418968200684]]], [[[1.0468040704727173]]], [[[1.4966363906860352]]], [[[1.1045222282409668]]], [[[1.800218105316162]]], [[[1.611971139907837]]], [[[1.7637742757797241]]], [[[1.231898546218872]]]], dtype='float32').reshape([11, 1, 1, 1]),
        ]


class TestPrimitiveOp_7d2e2d11697a73e5763287be4eb5691a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a1461f5dac00b621fb40b0f8d89877da
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[1.79573655128479]]], [[[1.817270278930664]]], [[[1.520437479019165]]], [[[1.0567792654037476]]], [[[1.0764904022216797]]], [[[1.572304606437683]]], [[[1.086213231086731]]], [[[1.461754322052002]]], [[[1.8199117183685303]]], [[[1.4717538356781006]]], [[[1.1371077299118042]]]], dtype='float32').reshape([11, 1, 1, 1]),
        ]


class TestPrimitiveOp_2add8be0f7b7ef5b92b6ae5b2887495d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a1461f5dac00b621fb40b0f8d89877da
    def get_inputs(self):
        return [
            paddle.uniform([43, 1, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_9ab45c754506e025a3d725f35c121574(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.floor(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 1, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_7b63ae5bbbf08697b75266b5b0b0b4d2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9ab45c754506e025a3d725f35c121574
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 8, 8], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ae11427357f360ef5bcb90aa21a4c864(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b47207d47005ed2d331b2c4c07e75725
    def get_inputs(self):
        return [
            paddle.uniform([5551, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_2add8be0f7b7ef5b92b6ae5b2887495d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a1461f5dac00b621fb40b0f8d89877da
    def get_inputs(self):
        return [
            paddle.uniform([43, 1, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_53d0799af372b148ed63b917989129e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b47207d47005ed2d331b2c4c07e75725
    def get_inputs(self):
        return [
            paddle.uniform([1769, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_649a7bc55a450636c2d7c786d7d855da(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b47207d47005ed2d331b2c4c07e75725
    def get_inputs(self):
        return [
            paddle.uniform([1502, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4ce2908666bf4ed4a3e36e4f479c40f5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a1461f5dac00b621fb40b0f8d89877da
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[1.4494937658309937]]], [[[1.2602022886276245]]], [[[1.1968384981155396]]], [[[1.9186632633209229]]], [[[1.228441596031189]]], [[[1.551252841949463]]], [[[1.5166237354278564]]], [[[1.0121961832046509]]], [[[1.7600574493408203]]], [[[1.804213523864746]]], [[[1.4629533290863037]]]], dtype='float32').reshape([11, 1, 1, 1]),
        ]


class TestPrimitiveOp_a6ce44bc6d25b8f3a2a79981a1aa7c65(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b47207d47005ed2d331b2c4c07e75725
    def get_inputs(self):
        return [
            paddle.uniform([2080, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_17935ea693801345740ce728003613d8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9ab45c754506e025a3d725f35c121574
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 64, 64], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_5de3b8fe779be8aa40d2940f6fd0ed21(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b47207d47005ed2d331b2c4c07e75725
    def get_inputs(self):
        return [
            paddle.uniform([4585, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_55d4709c564e380a86e85d148fd0fc96(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b47207d47005ed2d331b2c4c07e75725
    def get_inputs(self):
        return [
            paddle.uniform([1048, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_925f1ba8be857c1d091a694a056769a6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9ab45c754506e025a3d725f35c121574
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 128, 128], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_266055da53679fb21241d95b12b494ea(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b47207d47005ed2d331b2c4c07e75725
    def get_inputs(self):
        return [
            paddle.uniform([2390, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_30d28a0fa8aa56abcb08ad02f44459d8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b47207d47005ed2d331b2c4c07e75725
    def get_inputs(self):
        return [
            paddle.uniform([3090, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4de72a9c2dc19519b030baf1079bd82f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b47207d47005ed2d331b2c4c07e75725
    def get_inputs(self):
        return [
            paddle.uniform([3748, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_1237ec38a4635f586e986f09765b7446(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9ab45c754506e025a3d725f35c121574
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 16, 16], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_291b87ba9fe7441447eca051fcaf62c6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a1461f5dac00b621fb40b0f8d89877da
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[1.4360663890838623]]], [[[1.2438836097717285]]], [[[1.64339017868042]]], [[[1.2143800258636475]]], [[[1.6280211210250854]]], [[[1.377989649772644]]], [[[1.202406406402588]]], [[[1.0480618476867676]]], [[[1.8554184436798096]]], [[[1.5031912326812744]]], [[[1.8965623378753662]]]], dtype='float32').reshape([11, 1, 1, 1]),
        ]


class TestPrimitiveOp_2add8be0f7b7ef5b92b6ae5b2887495d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a1461f5dac00b621fb40b0f8d89877da
    def get_inputs(self):
        return [
            paddle.uniform([43, 1, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_1a4cd5f83de491281b206c87a77eeeac(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b47207d47005ed2d331b2c4c07e75725
    def get_inputs(self):
        return [
            paddle.uniform([2031, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_317efbfda08a2bab6adfcef3004ac5d3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9ab45c754506e025a3d725f35c121574
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 32, 32], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d4019122bfc6f52034ad4068bd5e5e0a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b47207d47005ed2d331b2c4c07e75725
    def get_inputs(self):
        return [
            paddle.uniform([4205, 4], dtype='float32', min=0, max=0.5),
        ]




if __name__ == '__main__':
    unittest.main()