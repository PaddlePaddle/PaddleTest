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


class TestPrimitiveOp_66cc45f25e57f5eb93d673a8bb2b2908(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b47207d47005ed2d331b2c4c07e75725
    def get_inputs(self):
        return [
            paddle.uniform([1508, 4], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_2f3a63d96a5a1c9c6aca444f2c41ec77(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b47207d47005ed2d331b2c4c07e75725
    def get_inputs(self):
        return [
            paddle.uniform([2377, 4], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_f1d3c8a7bd6d265a50ccd69e9d38698d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.floor(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[43, 1, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_719145c7764859229f6f0dc0346000fd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f1d3c8a7bd6d265a50ccd69e9d38698d
    def get_inputs(self):
        return [
            paddle.uniform([43, 1, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_719145c7764859229f6f0dc0346000fd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f1d3c8a7bd6d265a50ccd69e9d38698d
    def get_inputs(self):
        return [
            paddle.uniform([43, 1, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_719145c7764859229f6f0dc0346000fd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f1d3c8a7bd6d265a50ccd69e9d38698d
    def get_inputs(self):
        return [
            paddle.uniform([43, 1, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_5953982670548a16b679582e5dc86aad(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.floor(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[11, 1, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_d6413f116f96bdf511f3edd7294242df(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5953982670548a16b679582e5dc86aad
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[1.5291179418563843]]], [[[1.0183806419372559]]], [[[1.546187400817871]]], [[[1.2830346822738647]]], [[[1.6150920391082764]]], [[[1.2475271224975586]]], [[[1.4987990856170654]]], [[[0.968291699886322]]], [[[1.220901370048523]]], [[[1.377028226852417]]], [[[1.690088152885437]]]], dtype='float32').reshape([11, 1, 1, 1]),
        ]


class PrimitiveOp_e908836d1e84f9ad89506faa400c97e8(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.floor(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 1, 128, 128], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_5600481fa4f775bb556889a700716d14(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e908836d1e84f9ad89506faa400c97e8
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 128, 128], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_719145c7764859229f6f0dc0346000fd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f1d3c8a7bd6d265a50ccd69e9d38698d
    def get_inputs(self):
        return [
            paddle.uniform([43, 1, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_719145c7764859229f6f0dc0346000fd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f1d3c8a7bd6d265a50ccd69e9d38698d
    def get_inputs(self):
        return [
            paddle.uniform([43, 1, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_3a51808ce2f4aba40974b10fdaf63709(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b47207d47005ed2d331b2c4c07e75725
    def get_inputs(self):
        return [
            paddle.uniform([2015, 4], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_91e8c1e04fabb36f141f53dd2fee5ea9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5953982670548a16b679582e5dc86aad
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[1.512838363647461]]], [[[1.6476144790649414]]], [[[1.046582579612732]]], [[[1.4800596237182617]]], [[[1.6282541751861572]]], [[[0.984645426273346]]], [[[1.9003825187683105]]], [[[1.5627567768096924]]], [[[1.5293636322021484]]], [[[1.3458679914474487]]], [[[1.3245656490325928]]]], dtype='float32').reshape([11, 1, 1, 1]),
        ]


class TestPrimitiveOp_20cee9f0ab7833f4645b68957b52cbb2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b47207d47005ed2d331b2c4c07e75725
    def get_inputs(self):
        return [
            paddle.uniform([1830, 4], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_c0ae5503b274c4616d16fae015e1d8d0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b47207d47005ed2d331b2c4c07e75725
    def get_inputs(self):
        return [
            paddle.uniform([3039, 4], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_f838521de125abe599f1f645e7c30ae4(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.floor(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 1, 16, 16], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_c9cff21171437150affdd18bb9eb5fd2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f838521de125abe599f1f645e7c30ae4
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 16, 16], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_511317cc0e90cb834c2f9294930e46ba(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.floor(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 1, 64, 64], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_7f149393fa028287b8d4cd1172ecfee9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_511317cc0e90cb834c2f9294930e46ba
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 64, 64], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_e78a19842ed3e89f77791594ccde1c0f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.floor(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 1, 32, 32], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_1b047b873d01f22ee9c01d31e8ae3d5c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e78a19842ed3e89f77791594ccde1c0f
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 32, 32], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_54ad11509160eaaa10387c50f399e493(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5953982670548a16b679582e5dc86aad
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[0.997443675994873]]], [[[1.7239596843719482]]], [[[1.8344696760177612]]], [[[1.304682731628418]]], [[[1.5854151248931885]]], [[[1.6952823400497437]]], [[[1.162071704864502]]], [[[1.7472987174987793]]], [[[1.4802478551864624]]], [[[1.1451005935668945]]], [[[1.1422358751296997]]]], dtype='float32').reshape([11, 1, 1, 1]),
        ]


class TestPrimitiveOp_2d9b1c3e1f4e1d99ba7d3c8872c5bea8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b47207d47005ed2d331b2c4c07e75725
    def get_inputs(self):
        return [
            paddle.uniform([2046, 4], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_d6b6f270c29b66afe61b3f9ecfd06c15(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b47207d47005ed2d331b2c4c07e75725
    def get_inputs(self):
        return [
            paddle.uniform([5498, 4], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_cb3633609b0f31aac5c82c43cbf874af(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5953982670548a16b679582e5dc86aad
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[1.2966692447662354]]], [[[0.9863227009773254]]], [[[1.1734728813171387]]], [[[1.0552057027816772]]], [[[1.2254351377487183]]], [[[0.9833271503448486]]], [[[1.7648146152496338]]], [[[1.394010066986084]]], [[[1.261784315109253]]], [[[1.0235421657562256]]], [[[1.3297415971755981]]]], dtype='float32').reshape([11, 1, 1, 1]),
        ]


class TestPrimitiveOp_1c3f03f0c97bd50f7236292ca6a35118(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5953982670548a16b679582e5dc86aad
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[1.1402584314346313]]], [[[1.1630399227142334]]], [[[1.5274102687835693]]], [[[1.8526155948638916]]], [[[1.3933731317520142]]], [[[1.8664650917053223]]], [[[0.902289867401123]]], [[[0.9310075044631958]]], [[[1.4415092468261719]]], [[[1.4040696620941162]]], [[[1.0701242685317993]]]], dtype='float32').reshape([11, 1, 1, 1]),
        ]


class TestPrimitiveOp_8dda8bb8ea36b3473b965666f5ff8dd3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b47207d47005ed2d331b2c4c07e75725
    def get_inputs(self):
        return [
            paddle.uniform([1074, 4], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_a132dc35854309a5b20f02e4fa2664ba(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b47207d47005ed2d331b2c4c07e75725
    def get_inputs(self):
        return [
            paddle.uniform([1773, 4], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_fb16f3b9e74bb63693f72b9f5aa77a21(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.floor(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 1, 8, 8], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_cb2a3f7609dd4b67af06419be528ebda(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fb16f3b9e74bb63693f72b9f5aa77a21
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 8, 8], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_4cc786925b76bdeeff8b7ace72256597(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b47207d47005ed2d331b2c4c07e75725
    def get_inputs(self):
        return [
            paddle.uniform([4224, 4], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_3841aa89f79900b7925259e5f2ac0921(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b47207d47005ed2d331b2c4c07e75725
    def get_inputs(self):
        return [
            paddle.uniform([4657, 4], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_ff4867ff4b2e09ba7902fbae67337cb1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b47207d47005ed2d331b2c4c07e75725
    def get_inputs(self):
        return [
            paddle.uniform([3770, 4], dtype='float32', min=-0.5, max=0.5),
        ]




if __name__ == '__main__':
    unittest.main()