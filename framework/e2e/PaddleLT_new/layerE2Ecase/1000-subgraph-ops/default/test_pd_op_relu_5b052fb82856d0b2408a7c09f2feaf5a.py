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



class PrimitiveOp_d44a3f0cd996023fee4ee878a400af24(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 144, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_c85206e5e3b2db2a535514cbc941416e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d44a3f0cd996023fee4ee878a400af24
    def get_inputs(self):
        return [
            paddle.uniform([1, 144, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_166789628bf13f7b463f155dc0421d00(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 18], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_ff8e646d9e896fcc91e6eac6030b236e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_166789628bf13f7b463f155dc0421d00
    def get_inputs(self):
        return [
            paddle.to_tensor([[4.741256237030029, 4.269109725952148, 4.0213942527771, 4.1803669929504395, 4.374722957611084, 4.439574241638184, 4.903290271759033, 4.699925899505615, 4.434779644012451, 5.015420436859131, 3.9408249855041504, 3.8196098804473877, 3.7872872352600098, 5.112924575805664, 4.799351692199707, 4.1679277420043945, 4.076636791229248, 4.8087286949157715]], dtype='float32').reshape([1, 18]),
        ]


class PrimitiveOp_6267289b02ddf910490326ab0edd3459(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 23], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_6db1c5ffd3d9ee0e0692d2951f40f077(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6267289b02ddf910490326ab0edd3459
    def get_inputs(self):
        return [
            paddle.to_tensor([[5.130321025848389, 5.3894548416137695, 4.921603202819824, 5.491981029510498, 5.448110580444336, 5.147984027862549, 6.03997802734375, 5.238518714904785, 5.1570868492126465, 5.701995372772217, 5.314639568328857, 5.69205904006958, 5.123349189758301, 4.710868835449219, 5.074195861816406, 5.885203838348389, 5.568380355834961, 5.898579120635986, 5.344457626342773, 4.099438190460205, 5.689074993133545, 5.502340793609619, 5.345932960510254]], dtype='float32').reshape([1, 23]),
        ]


class PrimitiveOp_087f1861b31113457822ac08f97c3338(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 40, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_280d13cfc4843d055eba8d88a90105e2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_087f1861b31113457822ac08f97c3338
    def get_inputs(self):
        return [
            paddle.uniform([1, 40, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_4da04ec29fe5901e05a68968b1599868(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 240], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_5f3430df2ac8eba743c90d6d4136f423(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4da04ec29fe5901e05a68968b1599868
    def get_inputs(self):
        return [
            paddle.uniform([1, 240], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_fe587ef1b0275d833eef3dee9743136e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 120], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_ca678f59f8e56c6b730207805766e225(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe587ef1b0275d833eef3dee9743136e
    def get_inputs(self):
        return [
            paddle.uniform([1, 120], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_78aa65d9647b578c2f80bf46dde41aaa(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 4, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_69c639fdba93bfd607e23b8bed0ca4e4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_78aa65d9647b578c2f80bf46dde41aaa
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 20, 32], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_2a29f41474f242c7cc59958fcddacb2e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 1024], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_2d938651ccde0fc8197d90f6a2890b67(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a29f41474f242c7cc59958fcddacb2e
    def get_inputs(self):
        return [
            paddle.uniform([512, 1024], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_2d938651ccde0fc8197d90f6a2890b67(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a29f41474f242c7cc59958fcddacb2e
    def get_inputs(self):
        return [
            paddle.uniform([512, 1024], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_fb377a9795497c0bdea579bf8eeb09fe(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 168, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_2a572e355636ffe6b28ad3970ede1011(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fb377a9795497c0bdea579bf8eeb09fe
    def get_inputs(self):
        return [
            paddle.uniform([1, 168, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_e959870eeefa9ce0c62dc82cca569d99(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 30, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_75bef1f252be70024af78e0c788f88d5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e959870eeefa9ce0c62dc82cca569d99
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[7.599609375]], [[7.373504161834717]], [[6.686559200286865]], [[7.666210651397705]], [[6.684614658355713]], [[7.022700309753418]], [[7.361330032348633]], [[7.8938422203063965]], [[8.184746742248535]], [[7.0876569747924805]], [[7.430819511413574]], [[7.364805221557617]], [[7.855962753295898]], [[7.761816024780273]], [[6.407794952392578]], [[7.460008144378662]], [[7.458975791931152]], [[7.260522365570068]], [[6.946539878845215]], [[7.5184173583984375]], [[7.103298664093018]], [[7.72168493270874]], [[7.872062683105469]], [[7.041070938110352]], [[7.695741653442383]], [[8.167027473449707]], [[6.617100715637207]], [[7.137465476989746]], [[7.365726470947266]], [[7.626412391662598]]]], dtype='float32').reshape([1, 30, 1, 1]),
        ]


class PrimitiveOp_6286f031125a2ae84d98d4a84c8a69a3(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 84], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_d9472f3a223be7c3044d56d1c0e3b2a8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6286f031125a2ae84d98d4a84c8a69a3
    def get_inputs(self):
        return [
            paddle.uniform([10, 84], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_e606288976cfe89bc00ac47f1602a99e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 256, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_b831d693fb4535602e96c5a4a7af4ad1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e606288976cfe89bc00ac47f1602a99e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 80, 120], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b831d693fb4535602e96c5a4a7af4ad1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e606288976cfe89bc00ac47f1602a99e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 80, 120], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b831d693fb4535602e96c5a4a7af4ad1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e606288976cfe89bc00ac47f1602a99e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 80, 120], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b831d693fb4535602e96c5a4a7af4ad1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e606288976cfe89bc00ac47f1602a99e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 80, 120], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b831d693fb4535602e96c5a4a7af4ad1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e606288976cfe89bc00ac47f1602a99e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 80, 120], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b831d693fb4535602e96c5a4a7af4ad1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e606288976cfe89bc00ac47f1602a99e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 80, 120], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b831d693fb4535602e96c5a4a7af4ad1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e606288976cfe89bc00ac47f1602a99e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 80, 120], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b831d693fb4535602e96c5a4a7af4ad1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e606288976cfe89bc00ac47f1602a99e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 80, 120], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c64ec824693deb660caf2a987deaf7b2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e606288976cfe89bc00ac47f1602a99e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 40, 60], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c64ec824693deb660caf2a987deaf7b2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e606288976cfe89bc00ac47f1602a99e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 40, 60], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c64ec824693deb660caf2a987deaf7b2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e606288976cfe89bc00ac47f1602a99e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 40, 60], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c64ec824693deb660caf2a987deaf7b2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e606288976cfe89bc00ac47f1602a99e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 40, 60], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c64ec824693deb660caf2a987deaf7b2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e606288976cfe89bc00ac47f1602a99e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 40, 60], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c64ec824693deb660caf2a987deaf7b2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e606288976cfe89bc00ac47f1602a99e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 40, 60], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c64ec824693deb660caf2a987deaf7b2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e606288976cfe89bc00ac47f1602a99e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 40, 60], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c64ec824693deb660caf2a987deaf7b2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e606288976cfe89bc00ac47f1602a99e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 40, 60], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_04759c778161ff32ec76f980c8292aeb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e606288976cfe89bc00ac47f1602a99e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 20, 30], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_04759c778161ff32ec76f980c8292aeb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e606288976cfe89bc00ac47f1602a99e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 20, 30], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_04759c778161ff32ec76f980c8292aeb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e606288976cfe89bc00ac47f1602a99e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 20, 30], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_04759c778161ff32ec76f980c8292aeb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e606288976cfe89bc00ac47f1602a99e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 20, 30], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_04759c778161ff32ec76f980c8292aeb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e606288976cfe89bc00ac47f1602a99e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 20, 30], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_04759c778161ff32ec76f980c8292aeb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e606288976cfe89bc00ac47f1602a99e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 20, 30], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_04759c778161ff32ec76f980c8292aeb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e606288976cfe89bc00ac47f1602a99e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 20, 30], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_04759c778161ff32ec76f980c8292aeb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e606288976cfe89bc00ac47f1602a99e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 20, 30], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_1e230f287cabff66142752176cf2698d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e606288976cfe89bc00ac47f1602a99e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 10, 15], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_1e230f287cabff66142752176cf2698d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e606288976cfe89bc00ac47f1602a99e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 10, 15], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_1e230f287cabff66142752176cf2698d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e606288976cfe89bc00ac47f1602a99e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 10, 15], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_1e230f287cabff66142752176cf2698d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e606288976cfe89bc00ac47f1602a99e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 10, 15], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_1e230f287cabff66142752176cf2698d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e606288976cfe89bc00ac47f1602a99e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 10, 15], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_1e230f287cabff66142752176cf2698d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e606288976cfe89bc00ac47f1602a99e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 10, 15], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_1e230f287cabff66142752176cf2698d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e606288976cfe89bc00ac47f1602a99e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 10, 15], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_1e230f287cabff66142752176cf2698d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e606288976cfe89bc00ac47f1602a99e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 10, 15], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_91d0bab2f36ec69466402dfb99e03ec7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e606288976cfe89bc00ac47f1602a99e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 5, 8], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_91d0bab2f36ec69466402dfb99e03ec7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e606288976cfe89bc00ac47f1602a99e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 5, 8], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_91d0bab2f36ec69466402dfb99e03ec7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e606288976cfe89bc00ac47f1602a99e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 5, 8], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_91d0bab2f36ec69466402dfb99e03ec7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e606288976cfe89bc00ac47f1602a99e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 5, 8], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_91d0bab2f36ec69466402dfb99e03ec7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e606288976cfe89bc00ac47f1602a99e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 5, 8], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_91d0bab2f36ec69466402dfb99e03ec7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e606288976cfe89bc00ac47f1602a99e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 5, 8], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_91d0bab2f36ec69466402dfb99e03ec7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e606288976cfe89bc00ac47f1602a99e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 5, 8], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_91d0bab2f36ec69466402dfb99e03ec7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e606288976cfe89bc00ac47f1602a99e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 5, 8], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_97636085df8330b1b91fdef31ce0eb3f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e959870eeefa9ce0c62dc82cca569d99
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[7.9199042320251465]], [[7.963931560516357]], [[7.667877674102783]], [[7.773496150970459]], [[8.085932731628418]], [[8.820511817932129]], [[6.449216365814209]], [[7.2168869972229]], [[7.742691516876221]], [[8.471738815307617]], [[7.992523193359375]], [[6.955296993255615]], [[7.472898483276367]], [[7.46894645690918]], [[7.751184940338135]], [[7.649600028991699]], [[7.471212863922119]], [[7.898178577423096]], [[7.598336696624756]], [[8.48615550994873]], [[7.778166770935059]], [[7.871760368347168]], [[8.46120834350586]], [[8.784133911132812]], [[7.153819561004639]], [[7.055203914642334]], [[8.119718551635742]], [[8.323348045349121]], [[7.513210773468018]], [[8.193267822265625]]]], dtype='float32').reshape([1, 30, 1, 1]),
        ]


class TestPrimitiveOp_3c92c5e8883dac20df8840ceebbcfb13(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_78aa65d9647b578c2f80bf46dde41aaa
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 50, 76], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_2548c2832b93a7ba64cbccd164ac1953(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 5, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_376e6e5731b33ec84b3c2168e3fe0f34(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2548c2832b93a7ba64cbccd164ac1953
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[1.0951694250106812]], [[1.5151374340057373]], [[1.5183920860290527]], [[1.6139439344406128]], [[1.3948242664337158]]]], dtype='float32').reshape([1, 5, 1, 1]),
        ]


class PrimitiveOp_48439e7d4b3c00c0b373b599ee99c66f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 10, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_c3373f80ef66c0f34453ba9cd6850540(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_48439e7d4b3c00c0b373b599ee99c66f
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[2.8439249992370605]], [[2.516425848007202]], [[2.8137736320495605]], [[2.6823530197143555]], [[2.8902056217193604]], [[3.002372980117798]], [[3.297945261001587]], [[3.1611320972442627]], [[2.845583438873291]], [[2.604602813720703]]]], dtype='float32').reshape([1, 10, 1, 1]),
        ]


class PrimitiveOp_596dd1ada5fe26f97607c6e853778e25(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 240, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_92de2d354ff394a03d56f5a4bb353bef(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_596dd1ada5fe26f97607c6e853778e25
    def get_inputs(self):
        return [
            paddle.uniform([1, 240, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_0a358aee5a558727aee7da4d8b921119(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 24, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_ea1d8011975c93b6cf43ebf2b02e4bb8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0a358aee5a558727aee7da4d8b921119
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[5.982447147369385]], [[5.7862701416015625]], [[5.832577705383301]], [[5.829339027404785]], [[5.498556613922119]], [[6.5897536277771]], [[5.318362236022949]], [[5.776819229125977]], [[4.7671380043029785]], [[6.365700721740723]], [[6.029675006866455]], [[5.563591957092285]], [[5.445055961608887]], [[5.8314104080200195]], [[4.601117134094238]], [[5.939207553863525]], [[5.105357646942139]], [[6.047030925750732]], [[5.150845050811768]], [[6.351900100708008]], [[5.29506778717041]], [[5.765807151794434]], [[5.723557472229004]], [[5.554556846618652]]]], dtype='float32').reshape([1, 24, 1, 1]),
        ]


class TestPrimitiveOp_1a7f94ea1b6c1a22094bf2d8aa864540(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_78aa65d9647b578c2f80bf46dde41aaa
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 100, 152], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_efbc759c4c660603e9e3b0aac4e84795(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e606288976cfe89bc00ac47f1602a99e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 13, 19], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_1809f4b1d4ee44c948739241f0456a93(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 15], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_b9b6c2a8269180b52c3845b4bffc4c9c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1809f4b1d4ee44c948739241f0456a93
    def get_inputs(self):
        return [
            paddle.uniform([10, 15], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_e1b0ce1c40fb511cc63ef007b5b437a6(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 18, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_8ba4da3506db2033de81e420b544c7eb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e1b0ce1c40fb511cc63ef007b5b437a6
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[4.2925920486450195]], [[4.752250671386719]], [[4.5086236000061035]], [[4.488072395324707]], [[3.670379400253296]], [[5.065225601196289]], [[4.958056926727295]], [[4.283141136169434]], [[4.167104244232178]], [[4.248365879058838]], [[4.824505805969238]], [[4.317460536956787]], [[4.717772960662842]], [[5.272263050079346]], [[4.663010597229004]], [[4.814130783081055]], [[4.48759126663208]], [[4.700754642486572]]]], dtype='float32').reshape([1, 18, 1, 1]),
        ]


class TestPrimitiveOp_92de2d354ff394a03d56f5a4bb353bef(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_596dd1ada5fe26f97607c6e853778e25
    def get_inputs(self):
        return [
            paddle.uniform([1, 240, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_5e963e714ef26e0d4f8b7910fe928162(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0a358aee5a558727aee7da4d8b921119
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[6.276838302612305]], [[5.743196487426758]], [[5.426851272583008]], [[6.2739410400390625]], [[6.162806510925293]], [[5.822683811187744]], [[6.190535068511963]], [[5.62785005569458]], [[5.502087593078613]], [[5.353496551513672]], [[6.260344505310059]], [[6.771033763885498]], [[5.726195335388184]], [[6.176863193511963]], [[5.638212203979492]], [[6.01679801940918]], [[5.008150100708008]], [[5.82957124710083]], [[6.328047752380371]], [[6.707180976867676]], [[4.918158054351807]], [[5.786004543304443]], [[5.451436519622803]], [[6.393585205078125]]]], dtype='float32').reshape([1, 24, 1, 1]),
        ]


class TestPrimitiveOp_9b90ee56b8dc88ff8d10664db9c1cff4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6286f031125a2ae84d98d4a84c8a69a3
    def get_inputs(self):
        return [
            paddle.uniform([145, 84], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3c604c9cfdb1f70ebb9a6c83ba2e9b3f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_78aa65d9647b578c2f80bf46dde41aaa
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 28, 40], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_8d09468731d318e052363fc338f6df46(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 4, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_311d833d88bae2e5cc9466e402370ee8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8d09468731d318e052363fc338f6df46
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[0.9640469551086426]], [[1.1907650232315063]], [[0.7865208387374878]], [[1.3860007524490356]]]], dtype='float32').reshape([1, 4, 1, 1]),
        ]


class TestPrimitiveOp_9b90ee56b8dc88ff8d10664db9c1cff4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6286f031125a2ae84d98d4a84c8a69a3
    def get_inputs(self):
        return [
            paddle.uniform([145, 84], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_82b2c508f6c9f923272a3096d0beb108(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 11, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_8bf84180132a332d1d66583169d8ada9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_82b2c508f6c9f923272a3096d0beb108
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[3.1423606872558594]], [[2.7449698448181152]], [[2.6360058784484863]], [[2.6123995780944824]], [[2.776981830596924]], [[2.8372039794921875]], [[2.8127024173736572]], [[2.5624544620513916]], [[2.6382553577423096]], [[2.658640146255493]], [[2.701753616333008]]]], dtype='float32').reshape([1, 11, 1, 1]),
        ]


class TestPrimitiveOp_280d13cfc4843d055eba8d88a90105e2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_087f1861b31113457822ac08f97c3338
    def get_inputs(self):
        return [
            paddle.uniform([1, 40, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_92de2d354ff394a03d56f5a4bb353bef(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_596dd1ada5fe26f97607c6e853778e25
    def get_inputs(self):
        return [
            paddle.uniform([1, 240, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_e97529b02f651226bea345ddc53c376f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 120, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_dc8ea1d2e983d22632e0728730a19b98(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e97529b02f651226bea345ddc53c376f
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c3c7a0bbc22e5e0d4194acd6400f8173(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e959870eeefa9ce0c62dc82cca569d99
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[7.243147850036621]], [[7.573616027832031]], [[7.823995590209961]], [[7.255865573883057]], [[8.409427642822266]], [[7.9907331466674805]], [[7.504103660583496]], [[7.514348983764648]], [[8.195526123046875]], [[7.207747936248779]], [[7.370931148529053]], [[7.727114677429199]], [[7.150063514709473]], [[7.007374286651611]], [[7.798734188079834]], [[7.416893005371094]], [[6.929642677307129]], [[7.325921535491943]], [[7.392806053161621]], [[7.585785865783691]], [[7.384944438934326]], [[8.287138938903809]], [[7.234395980834961]], [[7.357141971588135]], [[6.452934265136719]], [[6.882917404174805]], [[7.055366039276123]], [[7.2592034339904785]], [[6.864374160766602]], [[7.8759870529174805]]]], dtype='float32').reshape([1, 30, 1, 1]),
        ]


class TestPrimitiveOp_2a572e355636ffe6b28ad3970ede1011(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fb377a9795497c0bdea579bf8eeb09fe
    def get_inputs(self):
        return [
            paddle.uniform([1, 168, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_c7a045eca5e06ec115dbbff400ba6662(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 60], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_e8640cb2ea7af9701834b1c055abfb91(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c7a045eca5e06ec115dbbff400ba6662
    def get_inputs(self):
        return [
            paddle.uniform([145, 60], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a43ea515e86190f5516c6d5e6ea429e9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_78aa65d9647b578c2f80bf46dde41aaa
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 80, 128], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_777c7160adaf882536ea2f53137f7dad(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 16, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_d78ab7d86cc3af8af66c69cb290527f6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_777c7160adaf882536ea2f53137f7dad
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[4.749521255493164]], [[4.21878719329834]], [[4.545407295227051]], [[4.051250457763672]], [[4.363979816436768]], [[4.789127826690674]], [[4.588782787322998]], [[4.309381008148193]], [[4.065494060516357]], [[4.195837497711182]], [[4.131467342376709]], [[4.5388970375061035]], [[4.305502891540527]], [[4.596755504608154]], [[4.036157608032227]], [[4.550198554992676]]]], dtype='float32').reshape([1, 16, 1, 1]),
        ]


class TestPrimitiveOp_ab6c4e97d8e3db540d11ae7d3542ccf1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_78aa65d9647b578c2f80bf46dde41aaa
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 14, 20], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9a72102ae1f0825abb5737bb71d5e063(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_78aa65d9647b578c2f80bf46dde41aaa
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 22, 33], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_8c8eef3e6e29c5a2f20121e79a2e04c4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_78aa65d9647b578c2f80bf46dde41aaa
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 23, 35], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_005bf68e0fe3057c2cb9f4ff884c11ce(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_78aa65d9647b578c2f80bf46dde41aaa
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 46, 70], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_2a572e355636ffe6b28ad3970ede1011(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fb377a9795497c0bdea579bf8eeb09fe
    def get_inputs(self):
        return [
            paddle.uniform([1, 168, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_fa9a609accde721e509c3469e230e74f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1809f4b1d4ee44c948739241f0456a93
    def get_inputs(self):
        return [
            paddle.uniform([22, 15], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_d2b6ec29c6007cab5cc9a49705a3b96e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 32, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_737c791a89c898fc04847f4ce0c9162b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d2b6ec29c6007cab5cc9a49705a3b96e
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_dc8ea1d2e983d22632e0728730a19b98(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e97529b02f651226bea345ddc53c376f
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a4a61fbc3c5657409906f6a0d5fdb104(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e959870eeefa9ce0c62dc82cca569d99
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[7.4858198165893555]], [[7.832398414611816]], [[7.306020259857178]], [[7.7038116455078125]], [[8.472077369689941]], [[7.497791290283203]], [[8.240334510803223]], [[8.199892044067383]], [[8.115281105041504]], [[7.499645709991455]], [[7.79252815246582]], [[7.45587158203125]], [[8.576510429382324]], [[6.370698928833008]], [[8.05994701385498]], [[7.457781791687012]], [[8.114330291748047]], [[7.668945789337158]], [[8.37739086151123]], [[7.923450946807861]], [[7.416363716125488]], [[8.187843322753906]], [[7.939794063568115]], [[7.338045597076416]], [[7.274503707885742]], [[7.770720481872559]], [[7.738190650939941]], [[7.896893501281738]], [[6.420858860015869]], [[6.9770283699035645]]]], dtype='float32').reshape([1, 30, 1, 1]),
        ]


class PrimitiveOp_a4dcbfeac1ca8c9f4347b7a2c270ab31(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 80, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_f0ecb0af7de4fd68cf8293df74f74e74(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a4dcbfeac1ca8c9f4347b7a2c270ab31
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_7390caf8c9c427d198ea94b5d5577c20(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 218], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_ffa996aa771fd424bcf6d4a92695faae(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7390caf8c9c427d198ea94b5d5577c20
    def get_inputs(self):
        return [
            paddle.uniform([1, 218], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_4617feab09e563b51a144da544da4870(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 25, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_0938924973cbc771477efd6878cca7ac(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4617feab09e563b51a144da544da4870
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[7.146759986877441]], [[7.080157279968262]], [[6.758172988891602]], [[7.047105312347412]], [[6.770845890045166]], [[6.778164386749268]], [[6.903726577758789]], [[7.041503429412842]], [[6.57687520980835]], [[7.236694812774658]], [[7.535022735595703]], [[6.67450475692749]], [[7.428268909454346]], [[7.629580497741699]], [[6.597499847412109]], [[7.139479160308838]], [[7.175117015838623]], [[7.065701007843018]], [[6.582592487335205]], [[6.432466506958008]], [[7.287783145904541]], [[7.4722514152526855]], [[7.3339362144470215]], [[6.703911781311035]], [[6.807648658752441]]]], dtype='float32').reshape([1, 25, 1, 1]),
        ]


class TestPrimitiveOp_92de2d354ff394a03d56f5a4bb353bef(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_596dd1ada5fe26f97607c6e853778e25
    def get_inputs(self):
        return [
            paddle.uniform([1, 240, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_80f17f8ade641ceae431cdc9b9441a8e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_78aa65d9647b578c2f80bf46dde41aaa
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 6, 9], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_2a572e355636ffe6b28ad3970ede1011(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fb377a9795497c0bdea579bf8eeb09fe
    def get_inputs(self):
        return [
            paddle.uniform([1, 168, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_8c66d2e1c7c97cb628931fef56a7d2a9(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 64, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_f2d80b357ca876e9500a54c7046eee56(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8c66d2e1c7c97cb628931fef56a7d2a9
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a851cc73ae436cf5646783c37622e4cd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a29f41474f242c7cc59958fcddacb2e
    def get_inputs(self):
        return [
            paddle.uniform([390, 1024], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a851cc73ae436cf5646783c37622e4cd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a29f41474f242c7cc59958fcddacb2e
    def get_inputs(self):
        return [
            paddle.uniform([390, 1024], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_8fac20c0e8951687c6a736e2756217e3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6286f031125a2ae84d98d4a84c8a69a3
    def get_inputs(self):
        return [
            paddle.uniform([171, 84], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_9c854ffaccea9cb8b32e40674d3127d3(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 60, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_6c2a9cd296804ce0f51d7ea2cef1506d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9c854ffaccea9cb8b32e40674d3127d3
    def get_inputs(self):
        return [
            paddle.uniform([1, 60, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_c9d1c2dd10643e5c51aa4bd2b0fdb4c7(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 20, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_6cca00687b87b36cda6a61ce5bce900b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c9d1c2dd10643e5c51aa4bd2b0fdb4c7
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[4.64856481552124]], [[5.657845973968506]], [[4.595557689666748]], [[4.741459369659424]], [[5.340050220489502]], [[5.148189544677734]], [[5.2109761238098145]], [[4.939141750335693]], [[5.3554511070251465]], [[4.626656532287598]], [[5.436121463775635]], [[5.367255210876465]], [[6.219812393188477]], [[5.431346893310547]], [[5.061332702636719]], [[5.448218822479248]], [[4.4503655433654785]], [[4.793201923370361]], [[4.995095729827881]], [[4.945746421813965]]]], dtype='float32').reshape([1, 20, 1, 1]),
        ]


class TestPrimitiveOp_2a572e355636ffe6b28ad3970ede1011(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fb377a9795497c0bdea579bf8eeb09fe
    def get_inputs(self):
        return [
            paddle.uniform([1, 168, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f0ecb0af7de4fd68cf8293df74f74e74(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a4dcbfeac1ca8c9f4347b7a2c270ab31
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_dc8ea1d2e983d22632e0728730a19b98(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e97529b02f651226bea345ddc53c376f
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_2a572e355636ffe6b28ad3970ede1011(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fb377a9795497c0bdea579bf8eeb09fe
    def get_inputs(self):
        return [
            paddle.uniform([1, 168, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_662392c3ab53871bca1d00dd6b408c55(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e1b0ce1c40fb511cc63ef007b5b437a6
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[4.799926280975342]], [[5.11075496673584]], [[4.990811347961426]], [[4.927366733551025]], [[4.920228004455566]], [[4.740077972412109]], [[4.691346645355225]], [[4.995629787445068]], [[5.1338653564453125]], [[4.486897945404053]], [[5.095236301422119]], [[4.979650020599365]], [[5.1125078201293945]], [[5.3938446044921875]], [[5.034809589385986]], [[4.830047607421875]], [[4.759293079376221]], [[5.464176654815674]]]], dtype='float32').reshape([1, 18, 1, 1]),
        ]


class TestPrimitiveOp_a4a61c944695c5f75b3502e90c6a2969(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c7a045eca5e06ec115dbbff400ba6662
    def get_inputs(self):
        return [
            paddle.uniform([10, 60], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c85206e5e3b2db2a535514cbc941416e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d44a3f0cd996023fee4ee878a400af24
    def get_inputs(self):
        return [
            paddle.uniform([1, 144, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_703a34c5df5adb595067bfbb91fcfe45(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_78aa65d9647b578c2f80bf46dde41aaa
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 7, 10], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_2a572e355636ffe6b28ad3970ede1011(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fb377a9795497c0bdea579bf8eeb09fe
    def get_inputs(self):
        return [
            paddle.uniform([1, 168, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_1774c9ccedef3922581dcc40dd7fe6be(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 96, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_bd8793acfac416c798c3d573dde0d971(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1774c9ccedef3922581dcc40dd7fe6be
    def get_inputs(self):
        return [
            paddle.uniform([43, 96, 109, 109], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_775603205c393624833ba935d4ef7239(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 16, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_1999dee0f4588dd6074ef0375dea1c7c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_775603205c393624833ba935d4ef7239
    def get_inputs(self):
        return [
            paddle.uniform([43, 16, 54, 54], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_081b0c899a155f8e8b9b92f30e16c3ce(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 64, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_380914a2477ff4c9f14983acd6e090e1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_081b0c899a155f8e8b9b92f30e16c3ce
    def get_inputs(self):
        return [
            paddle.uniform([43, 64, 54, 54], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_380914a2477ff4c9f14983acd6e090e1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_081b0c899a155f8e8b9b92f30e16c3ce
    def get_inputs(self):
        return [
            paddle.uniform([43, 64, 54, 54], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_1999dee0f4588dd6074ef0375dea1c7c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_775603205c393624833ba935d4ef7239
    def get_inputs(self):
        return [
            paddle.uniform([43, 16, 54, 54], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_380914a2477ff4c9f14983acd6e090e1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_081b0c899a155f8e8b9b92f30e16c3ce
    def get_inputs(self):
        return [
            paddle.uniform([43, 64, 54, 54], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_380914a2477ff4c9f14983acd6e090e1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_081b0c899a155f8e8b9b92f30e16c3ce
    def get_inputs(self):
        return [
            paddle.uniform([43, 64, 54, 54], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_017d29a1e4094c0e6532eff8c6d07cea(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 32, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_4386d5fa594b61bb25239f1f4c8083ef(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_017d29a1e4094c0e6532eff8c6d07cea
    def get_inputs(self):
        return [
            paddle.uniform([43, 32, 54, 54], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_e4adcd75dc29b0fdba55384f372344a0(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 128, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_2bd2f5151d11af3e1bc5b72ad3f1bcad(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e4adcd75dc29b0fdba55384f372344a0
    def get_inputs(self):
        return [
            paddle.uniform([43, 128, 54, 54], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_2bd2f5151d11af3e1bc5b72ad3f1bcad(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e4adcd75dc29b0fdba55384f372344a0
    def get_inputs(self):
        return [
            paddle.uniform([43, 128, 54, 54], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_43320452d583c32aca939e0ee6bc31db(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_017d29a1e4094c0e6532eff8c6d07cea
    def get_inputs(self):
        return [
            paddle.uniform([43, 32, 26, 26], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_fbd5c06872c3f8aa858d0f3f08396340(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e4adcd75dc29b0fdba55384f372344a0
    def get_inputs(self):
        return [
            paddle.uniform([43, 128, 26, 26], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_fbd5c06872c3f8aa858d0f3f08396340(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e4adcd75dc29b0fdba55384f372344a0
    def get_inputs(self):
        return [
            paddle.uniform([43, 128, 26, 26], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_531f1e2922208284d24d0824995301f7(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 48, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_7df9a965f629d239f8ee6913d0684d6e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_531f1e2922208284d24d0824995301f7
    def get_inputs(self):
        return [
            paddle.uniform([43, 48, 26, 26], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_9e25c2bd90b4a2216ed623d19f6b43fd(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 192, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_2a92ba55bcf93fef038f3458ab4885f8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9e25c2bd90b4a2216ed623d19f6b43fd
    def get_inputs(self):
        return [
            paddle.uniform([43, 192, 26, 26], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_2a92ba55bcf93fef038f3458ab4885f8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9e25c2bd90b4a2216ed623d19f6b43fd
    def get_inputs(self):
        return [
            paddle.uniform([43, 192, 26, 26], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_7df9a965f629d239f8ee6913d0684d6e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_531f1e2922208284d24d0824995301f7
    def get_inputs(self):
        return [
            paddle.uniform([43, 48, 26, 26], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_2a92ba55bcf93fef038f3458ab4885f8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9e25c2bd90b4a2216ed623d19f6b43fd
    def get_inputs(self):
        return [
            paddle.uniform([43, 192, 26, 26], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_2a92ba55bcf93fef038f3458ab4885f8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9e25c2bd90b4a2216ed623d19f6b43fd
    def get_inputs(self):
        return [
            paddle.uniform([43, 192, 26, 26], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_5b46255632cdb70190cd0de18dfcb779(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_081b0c899a155f8e8b9b92f30e16c3ce
    def get_inputs(self):
        return [
            paddle.uniform([43, 64, 26, 26], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e6ee8009f9e40096d4ff869835cc3f1e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e606288976cfe89bc00ac47f1602a99e
    def get_inputs(self):
        return [
            paddle.uniform([43, 256, 26, 26], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e6ee8009f9e40096d4ff869835cc3f1e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e606288976cfe89bc00ac47f1602a99e
    def get_inputs(self):
        return [
            paddle.uniform([43, 256, 26, 26], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4f12ae21b1777ebcd1d2b1935c14b597(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_081b0c899a155f8e8b9b92f30e16c3ce
    def get_inputs(self):
        return [
            paddle.uniform([43, 64, 12, 12], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_bd69b98510bfa2e2b234d414348497bd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e606288976cfe89bc00ac47f1602a99e
    def get_inputs(self):
        return [
            paddle.uniform([43, 256, 12, 12], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_bd69b98510bfa2e2b234d414348497bd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e606288976cfe89bc00ac47f1602a99e
    def get_inputs(self):
        return [
            paddle.uniform([43, 256, 12, 12], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_349a2e2d5ec0a758640b81356bde0d49(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 1000, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_a51231f0b52c0d59ed8b79f446eb0ec6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_349a2e2d5ec0a758640b81356bde0d49
    def get_inputs(self):
        return [
            paddle.uniform([43, 1000, 12, 12], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6c2a9cd296804ce0f51d7ea2cef1506d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9c854ffaccea9cb8b32e40674d3127d3
    def get_inputs(self):
        return [
            paddle.uniform([1, 60, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a2379bec54b3340fda57c1bdce289111(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e1b0ce1c40fb511cc63ef007b5b437a6
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[5.298534870147705]], [[5.194607734680176]], [[4.705836772918701]], [[4.816802501678467]], [[4.120375633239746]], [[4.8419189453125]], [[4.33341646194458]], [[4.572563171386719]], [[4.905956745147705]], [[4.852055549621582]], [[4.871097564697266]], [[4.523195266723633]], [[4.292860984802246]], [[4.8878889083862305]], [[4.615787506103516]], [[5.3019118309021]], [[4.301736354827881]], [[4.98100471496582]]]], dtype='float32').reshape([1, 18, 1, 1]),
        ]


class TestPrimitiveOp_d9472f3a223be7c3044d56d1c0e3b2a8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6286f031125a2ae84d98d4a84c8a69a3
    def get_inputs(self):
        return [
            paddle.uniform([10, 84], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_aba600d374d840010728e88a63706930(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0a358aee5a558727aee7da4d8b921119
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[6.101860046386719]], [[6.819554805755615]], [[7.243051528930664]], [[6.219446182250977]], [[6.800596714019775]], [[6.468642234802246]], [[7.288021564483643]], [[6.443670272827148]], [[5.773904323577881]], [[7.137299537658691]], [[5.9228434562683105]], [[6.262948513031006]], [[5.938857078552246]], [[6.216561317443848]], [[6.639321804046631]], [[6.653522968292236]], [[5.925400733947754]], [[6.456483840942383]], [[6.8000311851501465]], [[6.113401889801025]], [[6.260534763336182]], [[5.994530200958252]], [[7.381709575653076]], [[6.2493896484375]]]], dtype='float32').reshape([1, 24, 1, 1]),
        ]


class TestPrimitiveOp_458f993a24214c4368294dd5507630a6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e606288976cfe89bc00ac47f1602a99e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 11, 17], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_97f86771bec0de00b19cf5760c11e4c7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e1b0ce1c40fb511cc63ef007b5b437a6
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[4.906307697296143]], [[5.255023002624512]], [[4.802807807922363]], [[4.538781642913818]], [[4.59634256362915]], [[5.334607124328613]], [[5.358031749725342]], [[4.569863796234131]], [[4.9390058517456055]], [[4.65770149230957]], [[5.156092643737793]], [[4.605039119720459]], [[4.943058013916016]], [[5.026494026184082]], [[4.6596503257751465]], [[4.941858291625977]], [[5.574811935424805]], [[4.716567039489746]]]], dtype='float32').reshape([1, 18, 1, 1]),
        ]


class PrimitiveOp_c892b8f1985f28b7567cfa217ca2d883(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 48, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_95e609349fb34d09d4fda1cd10ec9754(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c892b8f1985f28b7567cfa217ca2d883
    def get_inputs(self):
        return [
            paddle.uniform([1, 48, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_dc2e67e37ff4872cd90894ec4e2fde1f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_78aa65d9647b578c2f80bf46dde41aaa
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 10, 16], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_5b5950ff2961296bfb8817569e05b21c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e1b0ce1c40fb511cc63ef007b5b437a6
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[4.111299991607666]], [[4.030351638793945]], [[4.914855480194092]], [[3.876230001449585]], [[4.093269348144531]], [[3.866751194000244]], [[3.9484663009643555]], [[3.648175001144409]], [[3.9560439586639404]], [[4.131312370300293]], [[3.6040196418762207]], [[4.455158233642578]], [[4.017114639282227]], [[4.072535037994385]], [[4.171624660491943]], [[4.0688300132751465]], [[4.168490886688232]], [[4.010280609130859]]]], dtype='float32').reshape([1, 18, 1, 1]),
        ]


class TestPrimitiveOp_95e609349fb34d09d4fda1cd10ec9754(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c892b8f1985f28b7567cfa217ca2d883
    def get_inputs(self):
        return [
            paddle.uniform([1, 48, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_71ff816cddef86f78e9562652b20f6c9(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 9], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_fd614fd824297bbbfcf182b2d45e5a1d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_71ff816cddef86f78e9562652b20f6c9
    def get_inputs(self):
        return [
            paddle.uniform([10, 9], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_99138ec5fcc58aaa6b9465293564d51c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_78aa65d9647b578c2f80bf46dde41aaa
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 12, 18], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_1015fec2c9df828a6c514ccf367f0ec9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1774c9ccedef3922581dcc40dd7fe6be
    def get_inputs(self):
        return [
            paddle.uniform([10, 96, 109, 109], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_290beec98dcdb72240718312386dcbf5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_775603205c393624833ba935d4ef7239
    def get_inputs(self):
        return [
            paddle.uniform([10, 16, 54, 54], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_64c23372c0d7c2c264794ef6695a89b2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_081b0c899a155f8e8b9b92f30e16c3ce
    def get_inputs(self):
        return [
            paddle.uniform([10, 64, 54, 54], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_64c23372c0d7c2c264794ef6695a89b2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_081b0c899a155f8e8b9b92f30e16c3ce
    def get_inputs(self):
        return [
            paddle.uniform([10, 64, 54, 54], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_290beec98dcdb72240718312386dcbf5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_775603205c393624833ba935d4ef7239
    def get_inputs(self):
        return [
            paddle.uniform([10, 16, 54, 54], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_64c23372c0d7c2c264794ef6695a89b2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_081b0c899a155f8e8b9b92f30e16c3ce
    def get_inputs(self):
        return [
            paddle.uniform([10, 64, 54, 54], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_64c23372c0d7c2c264794ef6695a89b2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_081b0c899a155f8e8b9b92f30e16c3ce
    def get_inputs(self):
        return [
            paddle.uniform([10, 64, 54, 54], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6d012ea8326271226e7aa550ce90c400(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_017d29a1e4094c0e6532eff8c6d07cea
    def get_inputs(self):
        return [
            paddle.uniform([10, 32, 54, 54], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_7fd1281a69a7e7923edc99e43f70ae75(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e4adcd75dc29b0fdba55384f372344a0
    def get_inputs(self):
        return [
            paddle.uniform([10, 128, 54, 54], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_7fd1281a69a7e7923edc99e43f70ae75(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e4adcd75dc29b0fdba55384f372344a0
    def get_inputs(self):
        return [
            paddle.uniform([10, 128, 54, 54], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b8438331259a940d8a8c2dc1a47ac50c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_017d29a1e4094c0e6532eff8c6d07cea
    def get_inputs(self):
        return [
            paddle.uniform([10, 32, 26, 26], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_2f98067cd191a07b5985bac2fe81dc2c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e4adcd75dc29b0fdba55384f372344a0
    def get_inputs(self):
        return [
            paddle.uniform([10, 128, 26, 26], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_2f98067cd191a07b5985bac2fe81dc2c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e4adcd75dc29b0fdba55384f372344a0
    def get_inputs(self):
        return [
            paddle.uniform([10, 128, 26, 26], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b11479232e105f0aa74b5b17a34d28d8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_531f1e2922208284d24d0824995301f7
    def get_inputs(self):
        return [
            paddle.uniform([10, 48, 26, 26], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9738bf94ae951e9e4841567e9de544b7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9e25c2bd90b4a2216ed623d19f6b43fd
    def get_inputs(self):
        return [
            paddle.uniform([10, 192, 26, 26], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9738bf94ae951e9e4841567e9de544b7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9e25c2bd90b4a2216ed623d19f6b43fd
    def get_inputs(self):
        return [
            paddle.uniform([10, 192, 26, 26], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b11479232e105f0aa74b5b17a34d28d8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_531f1e2922208284d24d0824995301f7
    def get_inputs(self):
        return [
            paddle.uniform([10, 48, 26, 26], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9738bf94ae951e9e4841567e9de544b7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9e25c2bd90b4a2216ed623d19f6b43fd
    def get_inputs(self):
        return [
            paddle.uniform([10, 192, 26, 26], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9738bf94ae951e9e4841567e9de544b7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9e25c2bd90b4a2216ed623d19f6b43fd
    def get_inputs(self):
        return [
            paddle.uniform([10, 192, 26, 26], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d6a20dcf856d09acf0426342ee657485(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_081b0c899a155f8e8b9b92f30e16c3ce
    def get_inputs(self):
        return [
            paddle.uniform([10, 64, 26, 26], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_dabc2005fa523c019186729422c01ff3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e606288976cfe89bc00ac47f1602a99e
    def get_inputs(self):
        return [
            paddle.uniform([10, 256, 26, 26], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_dabc2005fa523c019186729422c01ff3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e606288976cfe89bc00ac47f1602a99e
    def get_inputs(self):
        return [
            paddle.uniform([10, 256, 26, 26], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_425f7ba83a48a030cbbacf325b710d61(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_081b0c899a155f8e8b9b92f30e16c3ce
    def get_inputs(self):
        return [
            paddle.uniform([10, 64, 12, 12], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9b54449ec7e09d0667547e115198f4a8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e606288976cfe89bc00ac47f1602a99e
    def get_inputs(self):
        return [
            paddle.uniform([10, 256, 12, 12], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9b54449ec7e09d0667547e115198f4a8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e606288976cfe89bc00ac47f1602a99e
    def get_inputs(self):
        return [
            paddle.uniform([10, 256, 12, 12], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e71c5acdcec91a8a41e7d7c2aea83618(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_349a2e2d5ec0a758640b81356bde0d49
    def get_inputs(self):
        return [
            paddle.uniform([10, 1000, 12, 12], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_26809ba3759a413082b3c6cca954e47f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe587ef1b0275d833eef3dee9743136e
    def get_inputs(self):
        return [
            paddle.uniform([10, 120], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_737c791a89c898fc04847f4ce0c9162b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d2b6ec29c6007cab5cc9a49705a3b96e
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_bd03d402486b2231a1075cc25ac182b6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6286f031125a2ae84d98d4a84c8a69a3
    def get_inputs(self):
        return [
            paddle.uniform([22, 84], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_26c33aa8dc6db33785668608e1248346(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_78aa65d9647b578c2f80bf46dde41aaa
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 92, 140], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f254e52d466aa4171c890374089f2a4f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e606288976cfe89bc00ac47f1602a99e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 12, 18], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d7154a22fb6ff9e6dbeec3ddb3570680(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c7a045eca5e06ec115dbbff400ba6662
    def get_inputs(self):
        return [
            paddle.uniform([171, 60], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_8fac20c0e8951687c6a736e2756217e3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6286f031125a2ae84d98d4a84c8a69a3
    def get_inputs(self):
        return [
            paddle.uniform([171, 84], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_2bf53d8801d4e4af582a06a4ec8b18b7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_081b0c899a155f8e8b9b92f30e16c3ce
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 300, 300], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_2bf53d8801d4e4af582a06a4ec8b18b7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_081b0c899a155f8e8b9b92f30e16c3ce
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 300, 300], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_03d8673389296cbf40804582c9adf472(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e4adcd75dc29b0fdba55384f372344a0
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 150, 150], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_03d8673389296cbf40804582c9adf472(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e4adcd75dc29b0fdba55384f372344a0
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 150, 150], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_930db2674b200498b05031c0d9261ca3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e606288976cfe89bc00ac47f1602a99e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 75, 75], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_930db2674b200498b05031c0d9261ca3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e606288976cfe89bc00ac47f1602a99e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 75, 75], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_930db2674b200498b05031c0d9261ca3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e606288976cfe89bc00ac47f1602a99e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 75, 75], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_d240e935600cbe00c7acdfe5b776434e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 512, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_06d61b4a18f82be3b28bd7fc477f0abd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d240e935600cbe00c7acdfe5b776434e
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 38, 38], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_06d61b4a18f82be3b28bd7fc477f0abd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d240e935600cbe00c7acdfe5b776434e
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 38, 38], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_06d61b4a18f82be3b28bd7fc477f0abd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d240e935600cbe00c7acdfe5b776434e
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 38, 38], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_5b5e48b96a0707253da4fa4c19af5171(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d240e935600cbe00c7acdfe5b776434e
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 19, 19], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_5b5e48b96a0707253da4fa4c19af5171(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d240e935600cbe00c7acdfe5b776434e
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 19, 19], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_5b5e48b96a0707253da4fa4c19af5171(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d240e935600cbe00c7acdfe5b776434e
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 19, 19], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_b334a795cdbe453041551fd1e790ad03(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 1024, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_b204ce7e67ac94461b6c1625ccb87d54(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b334a795cdbe453041551fd1e790ad03
    def get_inputs(self):
        return [
            paddle.uniform([1, 1024, 19, 19], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b204ce7e67ac94461b6c1625ccb87d54(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b334a795cdbe453041551fd1e790ad03
    def get_inputs(self):
        return [
            paddle.uniform([1, 1024, 19, 19], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c5a5d66ec5d7451f4f7f84753e987bea(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e606288976cfe89bc00ac47f1602a99e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 19, 19], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c2983c35cc964a70d9df2000b05c3aa0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d240e935600cbe00c7acdfe5b776434e
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 10, 10], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_bf737a8e3c5c20739ea942a8b6811cf5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e4adcd75dc29b0fdba55384f372344a0
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 10, 10], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_abe0d0228d0fa8c1628de9a88f8f22c3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e606288976cfe89bc00ac47f1602a99e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 5, 5], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_736bdac3517165343a9c3796181402b4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e4adcd75dc29b0fdba55384f372344a0
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 5, 5], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c3303aa63e9147c2ea91cff733337390(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e606288976cfe89bc00ac47f1602a99e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_0a8d19796dbcaac8e6f1fdbfaf6bff8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e4adcd75dc29b0fdba55384f372344a0
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4bba1013ae8d0ab5752e0633417cf0ac(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e606288976cfe89bc00ac47f1602a99e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_2a572e355636ffe6b28ad3970ede1011(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fb377a9795497c0bdea579bf8eeb09fe
    def get_inputs(self):
        return [
            paddle.uniform([1, 168, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b2609ba1e0c4f7a09fe7f86452577e01(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_78aa65d9647b578c2f80bf46dde41aaa
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 13, 19], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4bbc3b58902fa97a1da9c684e07fef8f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1809f4b1d4ee44c948739241f0456a93
    def get_inputs(self):
        return [
            paddle.uniform([171, 15], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e7063ba9294d99e6b88364d218280cad(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c7a045eca5e06ec115dbbff400ba6662
    def get_inputs(self):
        return [
            paddle.uniform([22, 60], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6fa16da15d1ccd8877d96cceca67183f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_78aa65d9647b578c2f80bf46dde41aaa
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 25, 38], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e240a15d8b6c111941885405cd733af5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e1b0ce1c40fb511cc63ef007b5b437a6
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[5.330550193786621]], [[4.857364654541016]], [[4.123072624206543]], [[5.170295238494873]], [[4.986421585083008]], [[5.729411602020264]], [[4.739415168762207]], [[4.513397216796875]], [[5.018434047698975]], [[4.789000511169434]], [[4.771086692810059]], [[4.986555099487305]], [[4.491481304168701]], [[4.435744762420654]], [[5.35022497177124]], [[4.319518089294434]], [[5.210992336273193]], [[4.585388660430908]]]], dtype='float32').reshape([1, 18, 1, 1]),
        ]


class TestPrimitiveOp_6c2a9cd296804ce0f51d7ea2cef1506d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9c854ffaccea9cb8b32e40674d3127d3
    def get_inputs(self):
        return [
            paddle.uniform([1, 60, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_20a4d73604f3ae8631d5e3d20648df1e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 64, 13, 19], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_feb36fc04e96e6e91c29cbf8a6d028c6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20a4d73604f3ae8631d5e3d20648df1e
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 13, 19], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_280d13cfc4843d055eba8d88a90105e2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_087f1861b31113457822ac08f97c3338
    def get_inputs(self):
        return [
            paddle.uniform([1, 40, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_1fecdd2cbaf3343eace43a58a3a1e114(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e606288976cfe89bc00ac47f1602a99e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 16, 16], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ebb46f272814490263e6b58ab9b7f9cd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_777c7160adaf882536ea2f53137f7dad
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[4.744415283203125]], [[4.198853492736816]], [[4.303064823150635]], [[5.047729015350342]], [[4.643350601196289]], [[4.837965488433838]], [[4.654464244842529]], [[4.833505153656006]], [[5.067817687988281]], [[4.514011383056641]], [[4.704143047332764]], [[4.990448951721191]], [[4.468147277832031]], [[4.279181003570557]], [[4.4278950691223145]], [[5.0766754150390625]]]], dtype='float32').reshape([1, 16, 1, 1]),
        ]


class TestPrimitiveOp_628405f10f653e76a9011327ec0f40ae(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_71ff816cddef86f78e9562652b20f6c9
    def get_inputs(self):
        return [
            paddle.uniform([22, 9], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_dc8ea1d2e983d22632e0728730a19b98(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e97529b02f651226bea345ddc53c376f
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_8591e531a0e35f9636bdd9d4c4ecf2a5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e1b0ce1c40fb511cc63ef007b5b437a6
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[4.536120891571045]], [[4.536009311676025]], [[3.6040804386138916]], [[4.749693870544434]], [[4.702897071838379]], [[4.749359130859375]], [[4.354257106781006]], [[5.224194526672363]], [[4.8846282958984375]], [[4.1457200050354]], [[4.288418292999268]], [[4.426797866821289]], [[4.9749932289123535]], [[4.385109901428223]], [[4.084366321563721]], [[4.469963073730469]], [[4.016148567199707]], [[4.885502815246582]]]], dtype='float32').reshape([1, 18, 1, 1]),
        ]


class TestPrimitiveOp_88d98137a548d1635d2385cc381c3bdc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8d09468731d318e052363fc338f6df46
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[1.469990611076355]], [[1.1438980102539062]], [[1.0768877267837524]], [[1.7493946552276611]]]], dtype='float32').reshape([1, 4, 1, 1]),
        ]


class TestPrimitiveOp_fdf7f07ffe824c1c4ebbc4426db160d9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1774c9ccedef3922581dcc40dd7fe6be
    def get_inputs(self):
        return [
            paddle.uniform([11, 96, 109, 109], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_dd8010cdb87b0e3b1e8100f581182615(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_775603205c393624833ba935d4ef7239
    def get_inputs(self):
        return [
            paddle.uniform([11, 16, 54, 54], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3cc4989c1f3a6a09e7bd9921b792012b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_081b0c899a155f8e8b9b92f30e16c3ce
    def get_inputs(self):
        return [
            paddle.uniform([11, 64, 54, 54], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3cc4989c1f3a6a09e7bd9921b792012b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_081b0c899a155f8e8b9b92f30e16c3ce
    def get_inputs(self):
        return [
            paddle.uniform([11, 64, 54, 54], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_dd8010cdb87b0e3b1e8100f581182615(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_775603205c393624833ba935d4ef7239
    def get_inputs(self):
        return [
            paddle.uniform([11, 16, 54, 54], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3cc4989c1f3a6a09e7bd9921b792012b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_081b0c899a155f8e8b9b92f30e16c3ce
    def get_inputs(self):
        return [
            paddle.uniform([11, 64, 54, 54], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3cc4989c1f3a6a09e7bd9921b792012b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_081b0c899a155f8e8b9b92f30e16c3ce
    def get_inputs(self):
        return [
            paddle.uniform([11, 64, 54, 54], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_235c207e1d2330766d00439b66887a58(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_017d29a1e4094c0e6532eff8c6d07cea
    def get_inputs(self):
        return [
            paddle.uniform([11, 32, 54, 54], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_400680d649fc59c08d467bf8df354f5b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e4adcd75dc29b0fdba55384f372344a0
    def get_inputs(self):
        return [
            paddle.uniform([11, 128, 54, 54], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_400680d649fc59c08d467bf8df354f5b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e4adcd75dc29b0fdba55384f372344a0
    def get_inputs(self):
        return [
            paddle.uniform([11, 128, 54, 54], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e9048ba261c418abc0a8655800eb9fce(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_017d29a1e4094c0e6532eff8c6d07cea
    def get_inputs(self):
        return [
            paddle.uniform([11, 32, 26, 26], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9e942b0033050f59f45319472513a8e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e4adcd75dc29b0fdba55384f372344a0
    def get_inputs(self):
        return [
            paddle.uniform([11, 128, 26, 26], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9e942b0033050f59f45319472513a8e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e4adcd75dc29b0fdba55384f372344a0
    def get_inputs(self):
        return [
            paddle.uniform([11, 128, 26, 26], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_481eb58b859b2bc1ac875a9f4390a7d0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_531f1e2922208284d24d0824995301f7
    def get_inputs(self):
        return [
            paddle.uniform([11, 48, 26, 26], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_575a8107991623f080aa35058dbf1edc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9e25c2bd90b4a2216ed623d19f6b43fd
    def get_inputs(self):
        return [
            paddle.uniform([11, 192, 26, 26], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_575a8107991623f080aa35058dbf1edc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9e25c2bd90b4a2216ed623d19f6b43fd
    def get_inputs(self):
        return [
            paddle.uniform([11, 192, 26, 26], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_481eb58b859b2bc1ac875a9f4390a7d0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_531f1e2922208284d24d0824995301f7
    def get_inputs(self):
        return [
            paddle.uniform([11, 48, 26, 26], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_575a8107991623f080aa35058dbf1edc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9e25c2bd90b4a2216ed623d19f6b43fd
    def get_inputs(self):
        return [
            paddle.uniform([11, 192, 26, 26], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_575a8107991623f080aa35058dbf1edc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9e25c2bd90b4a2216ed623d19f6b43fd
    def get_inputs(self):
        return [
            paddle.uniform([11, 192, 26, 26], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9a7b322abf1617c640d34f7630559e4b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_081b0c899a155f8e8b9b92f30e16c3ce
    def get_inputs(self):
        return [
            paddle.uniform([11, 64, 26, 26], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_0a01966f4b8701981676cdca7428f02b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e606288976cfe89bc00ac47f1602a99e
    def get_inputs(self):
        return [
            paddle.uniform([11, 256, 26, 26], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_0a01966f4b8701981676cdca7428f02b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e606288976cfe89bc00ac47f1602a99e
    def get_inputs(self):
        return [
            paddle.uniform([11, 256, 26, 26], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_7799ba5a391b111c55a36d8bfa23d834(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_081b0c899a155f8e8b9b92f30e16c3ce
    def get_inputs(self):
        return [
            paddle.uniform([11, 64, 12, 12], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_db1be183c7a7f6f6e88131927e53e735(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e606288976cfe89bc00ac47f1602a99e
    def get_inputs(self):
        return [
            paddle.uniform([11, 256, 12, 12], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_db1be183c7a7f6f6e88131927e53e735(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e606288976cfe89bc00ac47f1602a99e
    def get_inputs(self):
        return [
            paddle.uniform([11, 256, 12, 12], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_232c7211a6ac1a16c907ed02a37f33ec(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_349a2e2d5ec0a758640b81356bde0d49
    def get_inputs(self):
        return [
            paddle.uniform([11, 1000, 12, 12], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6c2a9cd296804ce0f51d7ea2cef1506d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9c854ffaccea9cb8b32e40674d3127d3
    def get_inputs(self):
        return [
            paddle.uniform([1, 60, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f2d80b357ca876e9500a54c7046eee56(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8c66d2e1c7c97cb628931fef56a7d2a9
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_5aa86cc19fee5df8af52c0019ea77765(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1809f4b1d4ee44c948739241f0456a93
    def get_inputs(self):
        return [
            paddle.uniform([145, 15], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_751349000d6bee81c60d0aeba1912aae(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 168], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_60df5a415101214781bed3a9eecd4b30(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_751349000d6bee81c60d0aeba1912aae
    def get_inputs(self):
        return [
            paddle.uniform([1, 168], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_79f056f364567b2465724fd6a8e83a1c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 100, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_05dcf9e10fa8ef983b5600250e098b8f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_79f056f364567b2465724fd6a8e83a1c
    def get_inputs(self):
        return [
            paddle.uniform([1, 100, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_bd03d402486b2231a1075cc25ac182b6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6286f031125a2ae84d98d4a84c8a69a3
    def get_inputs(self):
        return [
            paddle.uniform([22, 84], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6c2a9cd296804ce0f51d7ea2cef1506d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9c854ffaccea9cb8b32e40674d3127d3
    def get_inputs(self):
        return [
            paddle.uniform([1, 60, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_dc8ea1d2e983d22632e0728730a19b98(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e97529b02f651226bea345ddc53c376f
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_95e609349fb34d09d4fda1cd10ec9754(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c892b8f1985f28b7567cfa217ca2d883
    def get_inputs(self):
        return [
            paddle.uniform([1, 48, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_2a572e355636ffe6b28ad3970ede1011(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fb377a9795497c0bdea579bf8eeb09fe
    def get_inputs(self):
        return [
            paddle.uniform([1, 168, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_528db60fbd998a6a364f3a0b5c302e67(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c9d1c2dd10643e5c51aa4bd2b0fdb4c7
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[5.039530277252197]], [[4.377685546875]], [[5.68862247467041]], [[5.335316181182861]], [[4.459051132202148]], [[5.4279303550720215]], [[5.450745105743408]], [[5.007411956787109]], [[4.7076568603515625]], [[5.455641269683838]], [[5.268977642059326]], [[5.0744194984436035]], [[5.373270034790039]], [[4.8659749031066895]], [[5.3133697509765625]], [[5.12187385559082]], [[4.553445816040039]], [[5.286235332489014]], [[5.7336812019348145]], [[5.181704044342041]]]], dtype='float32').reshape([1, 20, 1, 1]),
        ]


class PrimitiveOp_b06f3f66f48b1468907b194d0315299d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 84, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_27c92801d47ff9323f63973d0584b760(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b06f3f66f48b1468907b194d0315299d
    def get_inputs(self):
        return [
            paddle.uniform([1, 84, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_6bf861630a81723a2cec98b4fa32a9d3(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 12, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_574a425577d7158c20ffcbaafe1e94b5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6bf861630a81723a2cec98b4fa32a9d3
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[3.424633502960205]], [[3.643857955932617]], [[3.3726956844329834]], [[3.573767900466919]], [[4.213128566741943]], [[3.1926698684692383]], [[3.5392212867736816]], [[3.263819694519043]], [[3.622842788696289]], [[3.0635385513305664]], [[3.614889621734619]], [[3.183824300765991]]]], dtype='float32').reshape([1, 12, 1, 1]),
        ]


class TestPrimitiveOp_2d662645bd0be50bcee370190da2299c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c9d1c2dd10643e5c51aa4bd2b0fdb4c7
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[5.461369514465332]], [[5.290877342224121]], [[5.382344722747803]], [[5.661740303039551]], [[5.194095134735107]], [[5.612534046173096]], [[5.916957378387451]], [[5.657979488372803]], [[5.481818199157715]], [[5.548018932342529]], [[5.579656600952148]], [[4.979562282562256]], [[5.654099941253662]], [[5.421256065368652]], [[5.365314960479736]], [[5.782130718231201]], [[5.869709014892578]], [[5.4404497146606445]], [[5.646677017211914]], [[5.393651008605957]]]], dtype='float32').reshape([1, 20, 1, 1]),
        ]


class TestPrimitiveOp_e85e971e247391dd94ddc2acc9e57d11(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_82b2c508f6c9f923272a3096d0beb108
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[2.7602291107177734]], [[2.927309513092041]], [[2.4158554077148438]], [[2.3605599403381348]], [[3.0858049392700195]], [[3.0662009716033936]], [[2.486854076385498]], [[3.182710647583008]], [[2.8686933517456055]], [[2.5770089626312256]], [[3.3630216121673584]]]], dtype='float32').reshape([1, 11, 1, 1]),
        ]


class TestPrimitiveOp_dc8ea1d2e983d22632e0728730a19b98(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e97529b02f651226bea345ddc53c376f
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f254e52d466aa4171c890374089f2a4f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e606288976cfe89bc00ac47f1602a99e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 12, 18], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_05dcf9e10fa8ef983b5600250e098b8f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_79f056f364567b2465724fd6a8e83a1c
    def get_inputs(self):
        return [
            paddle.uniform([1, 100, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a7b5211e2107781a629c47c40a3f11f0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_78aa65d9647b578c2f80bf46dde41aaa
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 56, 80], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_010c79a5cb2ec4c93bd2b08a2c53d251(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 14, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_3ec5f245a506d85e13d3fedbd04e4059(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_010c79a5cb2ec4c93bd2b08a2c53d251
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[4.48289155960083]], [[4.444721221923828]], [[4.445754051208496]], [[4.0137176513671875]], [[3.732367753982544]], [[3.5434446334838867]], [[4.839659214019775]], [[3.154686689376831]], [[3.8090832233428955]], [[3.560391664505005]], [[3.637213945388794]], [[4.3086934089660645]], [[3.9956557750701904]], [[3.219186305999756]]]], dtype='float32').reshape([1, 14, 1, 1]),
        ]


class PrimitiveOp_6553b0c676e0a751f4fb6f9c2cdeec02(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 96, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_75d609ee63ac513d40511c773b69967a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6553b0c676e0a751f4fb6f9c2cdeec02
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_efbc759c4c660603e9e3b0aac4e84795(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e606288976cfe89bc00ac47f1602a99e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 13, 19], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_280d13cfc4843d055eba8d88a90105e2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_087f1861b31113457822ac08f97c3338
    def get_inputs(self):
        return [
            paddle.uniform([1, 40, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_2c78ac587364ee5f6aee5e3989a0d803(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c9d1c2dd10643e5c51aa4bd2b0fdb4c7
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[5.332091331481934]], [[5.385787487030029]], [[5.042260646820068]], [[5.451385021209717]], [[4.571070194244385]], [[4.791196346282959]], [[5.039742946624756]], [[4.820998668670654]], [[5.036779880523682]], [[5.543336868286133]], [[5.3662919998168945]], [[5.7566657066345215]], [[5.681779861450195]], [[5.677177429199219]], [[4.991026878356934]], [[5.15374755859375]], [[5.354227066040039]], [[5.846851348876953]], [[4.381585597991943]], [[5.291141986846924]]]], dtype='float32').reshape([1, 20, 1, 1]),
        ]


class TestPrimitiveOp_a99e4757012cc56263fc2d2d86c4aae1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0a358aee5a558727aee7da4d8b921119
    def get_inputs(self):
        return [
            paddle.uniform([2, 24, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a99e4757012cc56263fc2d2d86c4aae1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0a358aee5a558727aee7da4d8b921119
    def get_inputs(self):
        return [
            paddle.uniform([2, 24, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a99e4757012cc56263fc2d2d86c4aae1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0a358aee5a558727aee7da4d8b921119
    def get_inputs(self):
        return [
            paddle.uniform([2, 24, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a99e4757012cc56263fc2d2d86c4aae1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0a358aee5a558727aee7da4d8b921119
    def get_inputs(self):
        return [
            paddle.uniform([2, 24, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_99313ba4af9dea5fe586f9a06d4e7c29(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 6, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_5f75dcdaf88bf97fee70699d32c0e119(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_99313ba4af9dea5fe586f9a06d4e7c29
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[41449.71484375]], [[35008.82421875]], [[30557.6015625]], [[34631.3515625]], [[41193.93359375]], [[29123.91796875]]], [[[42809.31640625]], [[36154.83984375]], [[31563.177734375]], [[35762.07421875]], [[42543.3203125]], [[30078.4765625]]]], dtype='float32').reshape([2, 6, 1, 1]),
        ]


class TestPrimitiveOp_e291a777bbcd5ae0a9a7cf02a828a968(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_99313ba4af9dea5fe586f9a06d4e7c29
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[44367.31640625]], [[39087.90234375]], [[34349.6484375]], [[42728.58984375]], [[44984.35546875]], [[44794.37109375]]], [[[46859.09765625]], [[41273.64453125]], [[36279.625]], [[45118.44140625]], [[47505.70703125]], [[47309.5703125]]]], dtype='float32').reshape([2, 6, 1, 1]),
        ]


class TestPrimitiveOp_8ef405e53787006671ebca4ab893a7d7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_99313ba4af9dea5fe586f9a06d4e7c29
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[43127.96875]], [[40560.03515625]], [[47017.46484375]], [[40960.27734375]], [[36227.015625]], [[34119.4921875]]], [[[45698.82421875]], [[42976.1015625]], [[49825.80078125]], [[43406.92578125]], [[38385.984375]], [[36156.2421875]]]], dtype='float32').reshape([2, 6, 1, 1]),
        ]


class TestPrimitiveOp_d6cbef0bc9e3ee34bb868d9b0bc2b33c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_99313ba4af9dea5fe586f9a06d4e7c29
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[38161.83984375]], [[36818.5859375]], [[41175.0546875]], [[49611.91796875]], [[34411.4375]], [[42065.1953125]]], [[[40168.2890625]], [[38751.28515625]], [[43332.33203125]], [[52211.0078125]], [[36216.5703125]], [[44275.46484375]]]], dtype='float32').reshape([2, 6, 1, 1]),
        ]


class TestPrimitiveOp_ba459ffd1f83f46992fb28ba3ed144de(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e606288976cfe89bc00ac47f1602a99e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 96, 144], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ba459ffd1f83f46992fb28ba3ed144de(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e606288976cfe89bc00ac47f1602a99e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 96, 144], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ba459ffd1f83f46992fb28ba3ed144de(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e606288976cfe89bc00ac47f1602a99e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 96, 144], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ba459ffd1f83f46992fb28ba3ed144de(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e606288976cfe89bc00ac47f1602a99e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 96, 144], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ba459ffd1f83f46992fb28ba3ed144de(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e606288976cfe89bc00ac47f1602a99e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 96, 144], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ba459ffd1f83f46992fb28ba3ed144de(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e606288976cfe89bc00ac47f1602a99e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 96, 144], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ba459ffd1f83f46992fb28ba3ed144de(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e606288976cfe89bc00ac47f1602a99e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 96, 144], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ba459ffd1f83f46992fb28ba3ed144de(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e606288976cfe89bc00ac47f1602a99e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 96, 144], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c944f30a071b7ded742bb410d25ce534(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e606288976cfe89bc00ac47f1602a99e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 48, 72], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c944f30a071b7ded742bb410d25ce534(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e606288976cfe89bc00ac47f1602a99e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 48, 72], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c944f30a071b7ded742bb410d25ce534(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e606288976cfe89bc00ac47f1602a99e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 48, 72], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c944f30a071b7ded742bb410d25ce534(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e606288976cfe89bc00ac47f1602a99e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 48, 72], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c944f30a071b7ded742bb410d25ce534(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e606288976cfe89bc00ac47f1602a99e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 48, 72], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c944f30a071b7ded742bb410d25ce534(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e606288976cfe89bc00ac47f1602a99e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 48, 72], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c944f30a071b7ded742bb410d25ce534(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e606288976cfe89bc00ac47f1602a99e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 48, 72], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c944f30a071b7ded742bb410d25ce534(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e606288976cfe89bc00ac47f1602a99e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 48, 72], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_fecd58acdb05eef79b57345fdff1b59c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e606288976cfe89bc00ac47f1602a99e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 24, 36], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_fecd58acdb05eef79b57345fdff1b59c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e606288976cfe89bc00ac47f1602a99e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 24, 36], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_fecd58acdb05eef79b57345fdff1b59c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e606288976cfe89bc00ac47f1602a99e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 24, 36], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_fecd58acdb05eef79b57345fdff1b59c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e606288976cfe89bc00ac47f1602a99e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 24, 36], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_fecd58acdb05eef79b57345fdff1b59c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e606288976cfe89bc00ac47f1602a99e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 24, 36], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_fecd58acdb05eef79b57345fdff1b59c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e606288976cfe89bc00ac47f1602a99e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 24, 36], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_fecd58acdb05eef79b57345fdff1b59c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e606288976cfe89bc00ac47f1602a99e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 24, 36], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_fecd58acdb05eef79b57345fdff1b59c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e606288976cfe89bc00ac47f1602a99e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 24, 36], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f254e52d466aa4171c890374089f2a4f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e606288976cfe89bc00ac47f1602a99e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 12, 18], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f254e52d466aa4171c890374089f2a4f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e606288976cfe89bc00ac47f1602a99e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 12, 18], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f254e52d466aa4171c890374089f2a4f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e606288976cfe89bc00ac47f1602a99e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 12, 18], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f254e52d466aa4171c890374089f2a4f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e606288976cfe89bc00ac47f1602a99e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 12, 18], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f254e52d466aa4171c890374089f2a4f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e606288976cfe89bc00ac47f1602a99e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 12, 18], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f254e52d466aa4171c890374089f2a4f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e606288976cfe89bc00ac47f1602a99e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 12, 18], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f254e52d466aa4171c890374089f2a4f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e606288976cfe89bc00ac47f1602a99e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 12, 18], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f254e52d466aa4171c890374089f2a4f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e606288976cfe89bc00ac47f1602a99e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 12, 18], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_aa894ce9925afa7eb52f8780831dc616(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e606288976cfe89bc00ac47f1602a99e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 6, 9], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_aa894ce9925afa7eb52f8780831dc616(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e606288976cfe89bc00ac47f1602a99e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 6, 9], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_aa894ce9925afa7eb52f8780831dc616(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e606288976cfe89bc00ac47f1602a99e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 6, 9], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_aa894ce9925afa7eb52f8780831dc616(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e606288976cfe89bc00ac47f1602a99e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 6, 9], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_aa894ce9925afa7eb52f8780831dc616(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e606288976cfe89bc00ac47f1602a99e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 6, 9], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_aa894ce9925afa7eb52f8780831dc616(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e606288976cfe89bc00ac47f1602a99e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 6, 9], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_aa894ce9925afa7eb52f8780831dc616(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e606288976cfe89bc00ac47f1602a99e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 6, 9], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_aa894ce9925afa7eb52f8780831dc616(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e606288976cfe89bc00ac47f1602a99e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 6, 9], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_dc8ea1d2e983d22632e0728730a19b98(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e97529b02f651226bea345ddc53c376f
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_95e609349fb34d09d4fda1cd10ec9754(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c892b8f1985f28b7567cfa217ca2d883
    def get_inputs(self):
        return [
            paddle.uniform([1, 48, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_04370bdd3139db81b9a4444a3965cc41(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e959870eeefa9ce0c62dc82cca569d99
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[7.413014888763428]], [[8.107231140136719]], [[8.257291793823242]], [[7.519814491271973]], [[8.432050704956055]], [[8.355634689331055]], [[8.027410507202148]], [[7.123806953430176]], [[7.4549407958984375]], [[6.942090034484863]], [[7.500543117523193]], [[6.994082450866699]], [[7.175034046173096]], [[8.024166107177734]], [[7.7539896965026855]], [[7.590872287750244]], [[7.600566864013672]], [[7.318859100341797]], [[7.156850337982178]], [[6.473167419433594]], [[7.489564895629883]], [[8.77588939666748]], [[7.318371295928955]], [[7.173774242401123]], [[7.875977039337158]], [[8.524571418762207]], [[7.533117294311523]], [[7.948834419250488]], [[8.418747901916504]], [[7.8245649337768555]]]], dtype='float32').reshape([1, 30, 1, 1]),
        ]


class TestPrimitiveOp_464165625ec51a3005666cb123bff3c5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e959870eeefa9ce0c62dc82cca569d99
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[7.626097202301025]], [[7.268050193786621]], [[6.9301323890686035]], [[7.15386962890625]], [[8.064860343933105]], [[7.383313179016113]], [[7.4802985191345215]], [[7.5680084228515625]], [[7.965687274932861]], [[7.574403285980225]], [[8.067580223083496]], [[8.288963317871094]], [[7.9542236328125]], [[7.001071929931641]], [[8.693288803100586]], [[7.859289646148682]], [[8.046775817871094]], [[7.918967247009277]], [[8.335844039916992]], [[8.026721000671387]], [[8.32423210144043]], [[7.497042179107666]], [[7.500447750091553]], [[7.819343566894531]], [[7.2729692459106445]], [[7.537966251373291]], [[8.038290023803711]], [[7.229598045349121]], [[7.554515361785889]], [[7.194088935852051]]]], dtype='float32').reshape([1, 30, 1, 1]),
        ]


class TestPrimitiveOp_02ccb531ffa34ed191373856a125c834(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_78aa65d9647b578c2f80bf46dde41aaa
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 44, 66], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_07233f83629d11808f27a4a5d908dc7c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e959870eeefa9ce0c62dc82cca569d99
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[8.673772811889648]], [[8.158801078796387]], [[6.969350814819336]], [[7.467228412628174]], [[7.259758949279785]], [[8.396291732788086]], [[7.8252763748168945]], [[7.93901252746582]], [[8.612258911132812]], [[6.798539638519287]], [[8.076798439025879]], [[7.937400817871094]], [[7.692192554473877]], [[7.592185974121094]], [[8.022377967834473]], [[7.083656311035156]], [[8.478035926818848]], [[8.19157886505127]], [[8.37839126586914]], [[7.689326286315918]], [[8.248883247375488]], [[8.202616691589355]], [[9.069597244262695]], [[7.295444965362549]], [[7.947854995727539]], [[8.608205795288086]], [[7.111745357513428]], [[8.17653751373291]], [[7.840489864349365]], [[8.302075386047363]]]], dtype='float32').reshape([1, 30, 1, 1]),
        ]


class PrimitiveOp_99ac09e579e7234a2c35cc9891811417(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 50, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_92d5f7991565c0ffba701eaff8d4bdef(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_99ac09e579e7234a2c35cc9891811417
    def get_inputs(self):
        return [
            paddle.uniform([1, 50, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6c2a9cd296804ce0f51d7ea2cef1506d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9c854ffaccea9cb8b32e40674d3127d3
    def get_inputs(self):
        return [
            paddle.uniform([1, 60, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_0e9c44529cd0720e5df4775bb9188028(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e959870eeefa9ce0c62dc82cca569d99
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[8.187416076660156]], [[6.663419246673584]], [[7.9976396560668945]], [[7.31292724609375]], [[7.175165176391602]], [[6.976984977722168]], [[7.7108473777771]], [[7.995647430419922]], [[7.8040266036987305]], [[8.59719467163086]], [[7.5217509269714355]], [[8.143982887268066]], [[7.239157199859619]], [[7.5660552978515625]], [[8.412611961364746]], [[6.772968769073486]], [[7.627128601074219]], [[7.961509704589844]], [[8.316770553588867]], [[8.19364070892334]], [[8.135167121887207]], [[7.383008003234863]], [[8.779121398925781]], [[7.557380199432373]], [[7.348526477813721]], [[7.821513652801514]], [[7.1406073570251465]], [[7.949775218963623]], [[7.784019470214844]], [[8.507400512695312]]]], dtype='float32').reshape([1, 30, 1, 1]),
        ]


class TestPrimitiveOp_a6d756c9e3a782b2134672ea3d57356b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6bf861630a81723a2cec98b4fa32a9d3
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[3.3816990852355957]], [[3.742743968963623]], [[3.869838237762451]], [[2.9714837074279785]], [[3.5777106285095215]], [[3.6011648178100586]], [[3.4669716358184814]], [[3.324681043624878]], [[3.6677181720733643]], [[3.698624610900879]], [[3.4260833263397217]], [[2.9585728645324707]]]], dtype='float32').reshape([1, 12, 1, 1]),
        ]


class TestPrimitiveOp_16733302e3f9b0f2d070789d556914bd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6bf861630a81723a2cec98b4fa32a9d3
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[3.381941318511963]], [[3.397540807723999]], [[3.0476503372192383]], [[3.218015193939209]], [[2.9947872161865234]], [[3.041104316711426]], [[3.421494722366333]], [[2.895808696746826]], [[2.8716137409210205]], [[3.1972920894622803]], [[3.1228690147399902]], [[3.288539171218872]]]], dtype='float32').reshape([1, 12, 1, 1]),
        ]


class TestPrimitiveOp_69d8e0c18fb152ca84ddd5c80b4902d4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4617feab09e563b51a144da544da4870
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[6.450954437255859]], [[6.529915809631348]], [[6.740278244018555]], [[6.08266019821167]], [[6.230340003967285]], [[6.245777606964111]], [[5.4485578536987305]], [[6.26106071472168]], [[6.920781135559082]], [[6.531062126159668]], [[6.552233695983887]], [[7.047784328460693]], [[6.183830261230469]], [[6.331979751586914]], [[6.389163494110107]], [[6.6392130851745605]], [[7.358436584472656]], [[5.888974666595459]], [[6.789358139038086]], [[6.667937755584717]], [[6.902990341186523]], [[5.783409118652344]], [[6.549851894378662]], [[6.295942306518555]], [[6.4413628578186035]]]], dtype='float32').reshape([1, 25, 1, 1]),
        ]


class PrimitiveOp_94fa2d9f0e4164dc0e384b4c316906cc(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 72, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_d62e6d97418e6c563a1730b131af3b54(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_94fa2d9f0e4164dc0e384b4c316906cc
    def get_inputs(self):
        return [
            paddle.uniform([1, 72, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_656040d125d6a43e9bed89c9cbf12737(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 312], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_b98f3e728c26e47c8605181b18bbe91c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_656040d125d6a43e9bed89c9cbf12737
    def get_inputs(self):
        return [
            paddle.uniform([1, 312], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_af024c9d8c827be28357c56bff141677(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe587ef1b0275d833eef3dee9743136e
    def get_inputs(self):
        return [
            paddle.uniform([171, 120], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_992a9b7f851c61272e2e22c986449d1b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_71ff816cddef86f78e9562652b20f6c9
    def get_inputs(self):
        return [
            paddle.uniform([145, 9], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a48baf5b8a8a8f783060793cf771431f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_78aa65d9647b578c2f80bf46dde41aaa
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 5, 8], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_7dce3490cef9cad225ced9991043f938(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e1b0ce1c40fb511cc63ef007b5b437a6
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[3.8954665660858154]], [[4.304795742034912]], [[5.0634684562683105]], [[4.979977130889893]], [[4.358904838562012]], [[4.681210517883301]], [[4.717888832092285]], [[4.327868461608887]], [[4.89727783203125]], [[4.974392414093018]], [[4.489230632781982]], [[4.506154537200928]], [[4.661153316497803]], [[4.72791862487793]], [[4.687276840209961]], [[4.261350154876709]], [[4.265027046203613]], [[4.335094451904297]]]], dtype='float32').reshape([1, 18, 1, 1]),
        ]


class PrimitiveOp_a20ad9624fd99b6bc1260510d6768493(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 39], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_7b4088d529bd07ceb182e79c60a24105(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a20ad9624fd99b6bc1260510d6768493
    def get_inputs(self):
        return [
            paddle.uniform([1, 39], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_499d755cae0f78215e2887ab20a9d14f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 5, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_57dcf5d4a616b0614cb077761c19e911(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_499d755cae0f78215e2887ab20a9d14f
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[1.825123906135559]], [[1.860487937927246]], [[1.6420276165008545]], [[1.7224886417388916]], [[1.782343864440918]]]], dtype='float32').reshape([1, 5, 1, 1]),
        ]


class PrimitiveOp_1d53126bad4d80a51643d8a61015334a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 10, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_3a1c72c2b7f4f3a6504be768fb0e4ecb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1d53126bad4d80a51643d8a61015334a
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[3.838575839996338]], [[3.232434034347534]], [[2.656240463256836]], [[3.2078402042388916]], [[3.6649234294891357]], [[4.246079921722412]], [[2.986668109893799]], [[3.1966238021850586]], [[3.2245218753814697]], [[3.440957546234131]]]], dtype='float32').reshape([1, 10, 1, 1]),
        ]


class PrimitiveOp_3f8df330c316688faa19f9217bfcc107(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 20, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_f09018deed9bbbcdc23f92df85211bce(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3f8df330c316688faa19f9217bfcc107
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[5.263118267059326]], [[5.076326847076416]], [[4.819751262664795]], [[4.882174015045166]], [[4.812960624694824]], [[5.059957027435303]], [[5.180749893188477]], [[4.604582786560059]], [[5.390031337738037]], [[4.4930500984191895]], [[4.993330955505371]], [[4.644032955169678]], [[4.822340488433838]], [[4.3817925453186035]], [[4.718923568725586]], [[4.871263027191162]], [[5.316429138183594]], [[5.180314064025879]], [[5.470968246459961]], [[5.097471714019775]]]], dtype='float32').reshape([1, 20, 1, 1]),
        ]


class PrimitiveOp_91336aa18f721f51a7f84ddd5447385e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 40, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_d0743ae4cbb2c00f45f36567df573f81(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_91336aa18f721f51a7f84ddd5447385e
    def get_inputs(self):
        return [
            paddle.uniform([1, 40, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_92d5f7991565c0ffba701eaff8d4bdef(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_99ac09e579e7234a2c35cc9891811417
    def get_inputs(self):
        return [
            paddle.uniform([1, 50, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f0ecb0af7de4fd68cf8293df74f74e74(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a4dcbfeac1ca8c9f4347b7a2c270ab31
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_1e230f287cabff66142752176cf2698d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e606288976cfe89bc00ac47f1602a99e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 10, 15], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_dc8ea1d2e983d22632e0728730a19b98(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e97529b02f651226bea345ddc53c376f
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ffa996aa771fd424bcf6d4a92695faae(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7390caf8c9c427d198ea94b5d5577c20
    def get_inputs(self):
        return [
            paddle.uniform([1, 218], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6e68c11d7ff1406b1c3d1bf3737d8b34(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0a358aee5a558727aee7da4d8b921119
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[6.908782005310059]], [[6.370944976806641]], [[6.305734157562256]], [[6.727738857269287]], [[6.980925559997559]], [[6.699498176574707]], [[6.459497928619385]], [[6.577095031738281]], [[7.904592990875244]], [[6.11961030960083]], [[6.198531627655029]], [[6.923077583312988]], [[6.083276748657227]], [[5.65005350112915]], [[6.276230335235596]], [[6.18086576461792]], [[7.109280586242676]], [[6.330440521240234]], [[7.080116271972656]], [[6.102603912353516]], [[6.841432571411133]], [[6.064248085021973]], [[6.296365737915039]], [[6.250420093536377]]]], dtype='float32').reshape([1, 24, 1, 1]),
        ]


class TestPrimitiveOp_f0c5dc80dd59932a265bf976b34a6635(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe587ef1b0275d833eef3dee9743136e
    def get_inputs(self):
        return [
            paddle.uniform([22, 120], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_30ee9a3a1a2169d66f697cd055ad6dff(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_48439e7d4b3c00c0b373b599ee99c66f
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[2.9680581092834473]], [[2.671557903289795]], [[2.8932790756225586]], [[2.473017454147339]], [[2.941282033920288]], [[3.002377986907959]], [[3.4186253547668457]], [[3.0199623107910156]], [[3.2422003746032715]], [[2.7163093090057373]]]], dtype='float32').reshape([1, 10, 1, 1]),
        ]


class TestPrimitiveOp_e0239e21fd7c1b7344dc208d58f9a7d9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe587ef1b0275d833eef3dee9743136e
    def get_inputs(self):
        return [
            paddle.uniform([145, 120], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a640fa72be7d9696796030b3010178e6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_78aa65d9647b578c2f80bf46dde41aaa
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 40, 64], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_e6749159b9fb6b42facb623e6f2eb5a5(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 64, 50, 76], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_a5f8840f46bd15eacc2900bd802e4ae0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e6749159b9fb6b42facb623e6f2eb5a5
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 50, 76], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b98ada4ea62150cf8f8c63109553479d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_71ff816cddef86f78e9562652b20f6c9
    def get_inputs(self):
        return [
            paddle.uniform([171, 9], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_dc8ea1d2e983d22632e0728730a19b98(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e97529b02f651226bea345ddc53c376f
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_5252dc14745d02e7e192505e054b288b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e1b0ce1c40fb511cc63ef007b5b437a6
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[4.617293834686279]], [[4.896233558654785]], [[4.9815568923950195]], [[4.362690448760986]], [[4.823935031890869]], [[4.940952777862549]], [[5.3251142501831055]], [[5.107237815856934]], [[4.368494987487793]], [[3.9635629653930664]], [[4.179454326629639]], [[4.730003356933594]], [[5.343450546264648]], [[4.466681957244873]], [[5.378194332122803]], [[4.693898677825928]], [[4.804242134094238]], [[4.789822578430176]]]], dtype='float32').reshape([1, 18, 1, 1]),
        ]


class PrimitiveOp_c2ac2885ac4a8ce914e5c4d83342d316(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 30], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_5459fab638d7f382dedf9fb708804f54(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c2ac2885ac4a8ce914e5c4d83342d316
    def get_inputs(self):
        return [
            paddle.to_tensor([[7.777254104614258, 7.340043067932129, 7.899073123931885, 7.603011131286621, 8.432877540588379, 7.752668857574463, 7.35352087020874, 7.88749361038208, 8.87308120727539, 7.3297929763793945, 7.800246715545654, 7.732612609863281, 8.434696197509766, 7.713261127471924, 8.252337455749512, 7.998571395874023, 7.595489025115967, 7.319191932678223, 8.093716621398926, 8.356228828430176, 8.674049377441406, 7.128657341003418, 7.968542575836182, 8.350767135620117, 8.166156768798828, 8.414031982421875, 7.9140496253967285, 7.590310573577881, 7.526113033294678, 7.965360164642334]], dtype='float32').reshape([1, 30]),
        ]


class TestPrimitiveOp_2a572e355636ffe6b28ad3970ede1011(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fb377a9795497c0bdea579bf8eeb09fe
    def get_inputs(self):
        return [
            paddle.uniform([1, 168, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_2a572e355636ffe6b28ad3970ede1011(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fb377a9795497c0bdea579bf8eeb09fe
    def get_inputs(self):
        return [
            paddle.uniform([1, 168, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_60df5a415101214781bed3a9eecd4b30(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_751349000d6bee81c60d0aeba1912aae
    def get_inputs(self):
        return [
            paddle.uniform([1, 168], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4068f427d7ef056b625672afc9e96e11(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e959870eeefa9ce0c62dc82cca569d99
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[9.099907875061035]], [[8.808845520019531]], [[8.94848346710205]], [[8.904350280761719]], [[7.843731880187988]], [[8.555520057678223]], [[9.616324424743652]], [[8.678474426269531]], [[8.664719581604004]], [[9.420342445373535]], [[8.583364486694336]], [[9.24262523651123]], [[8.194258689880371]], [[9.038471221923828]], [[8.729104042053223]], [[7.5928168296813965]], [[9.081381797790527]], [[7.6608405113220215]], [[8.26392936706543]], [[8.710794448852539]], [[8.89842414855957]], [[8.832663536071777]], [[8.902417182922363]], [[8.453612327575684]], [[8.349315643310547]], [[9.200179100036621]], [[9.425411224365234]], [[9.194620132446289]], [[8.126853942871094]], [[9.308067321777344]]]], dtype='float32').reshape([1, 30, 1, 1]),
        ]


class TestPrimitiveOp_ffc2da21dcabb5c116594f08caeccf5b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2548c2832b93a7ba64cbccd164ac1953
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[1.7731372117996216]], [[1.6782431602478027]], [[1.3857113122940063]], [[1.0903266668319702]], [[1.584107518196106]]]], dtype='float32').reshape([1, 5, 1, 1]),
        ]


class TestPrimitiveOp_bbec4d04f018c168fc64ce1a4e262eb2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_48439e7d4b3c00c0b373b599ee99c66f
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[2.265617847442627]], [[2.509380578994751]], [[2.1786749362945557]], [[2.5498948097229004]], [[2.002563953399658]], [[2.137655735015869]], [[2.2031662464141846]], [[2.507338762283325]], [[2.6661641597747803]], [[1.9737988710403442]]]], dtype='float32').reshape([1, 10, 1, 1]),
        ]


class TestPrimitiveOp_761996b075dfb799697ca1ce10e3464f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c9d1c2dd10643e5c51aa4bd2b0fdb4c7
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[4.867809772491455]], [[5.040612697601318]], [[4.492854595184326]], [[4.457716941833496]], [[5.531546115875244]], [[4.842069149017334]], [[3.783423662185669]], [[4.4237775802612305]], [[4.8386712074279785]], [[5.009635925292969]], [[5.0376715660095215]], [[6.0887956619262695]], [[4.601489543914795]], [[5.349428653717041]], [[5.86975622177124]], [[5.116644382476807]], [[4.787876129150391]], [[5.059450626373291]], [[4.887513160705566]], [[5.024298191070557]]]], dtype='float32').reshape([1, 20, 1, 1]),
        ]


class TestPrimitiveOp_280d13cfc4843d055eba8d88a90105e2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_087f1861b31113457822ac08f97c3338
    def get_inputs(self):
        return [
            paddle.uniform([1, 40, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b5447469338291f0b3b170707ed0e82c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_777c7160adaf882536ea2f53137f7dad
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[4.288623332977295]], [[4.848727703094482]], [[3.858452558517456]], [[3.8020455837249756]], [[4.621124744415283]], [[4.233011245727539]], [[4.626319885253906]], [[3.9987354278564453]], [[3.6208770275115967]], [[3.79485821723938]], [[3.8595187664031982]], [[4.8308892250061035]], [[3.860363483428955]], [[3.2855732440948486]], [[4.631810665130615]], [[4.691732406616211]]]], dtype='float32').reshape([1, 16, 1, 1]),
        ]


class TestPrimitiveOp_737c791a89c898fc04847f4ce0c9162b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d2b6ec29c6007cab5cc9a49705a3b96e
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_65a7d4cc877c28cdc806b0207d0c244f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 36, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_bf9afed14dfe99d3c5fc94eb976b9aac(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_65a7d4cc877c28cdc806b0207d0c244f
    def get_inputs(self):
        return [
            paddle.uniform([1, 36, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ece1e73dc270237d8a3d1951922a630c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e606288976cfe89bc00ac47f1602a99e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 92, 140], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ece1e73dc270237d8a3d1951922a630c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e606288976cfe89bc00ac47f1602a99e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 92, 140], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ece1e73dc270237d8a3d1951922a630c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e606288976cfe89bc00ac47f1602a99e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 92, 140], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ece1e73dc270237d8a3d1951922a630c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e606288976cfe89bc00ac47f1602a99e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 92, 140], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ece1e73dc270237d8a3d1951922a630c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e606288976cfe89bc00ac47f1602a99e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 92, 140], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ece1e73dc270237d8a3d1951922a630c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e606288976cfe89bc00ac47f1602a99e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 92, 140], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ece1e73dc270237d8a3d1951922a630c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e606288976cfe89bc00ac47f1602a99e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 92, 140], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ece1e73dc270237d8a3d1951922a630c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e606288976cfe89bc00ac47f1602a99e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 92, 140], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d89e84045749a23ea4d0eed6696846a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e606288976cfe89bc00ac47f1602a99e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 46, 70], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d89e84045749a23ea4d0eed6696846a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e606288976cfe89bc00ac47f1602a99e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 46, 70], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d89e84045749a23ea4d0eed6696846a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e606288976cfe89bc00ac47f1602a99e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 46, 70], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d89e84045749a23ea4d0eed6696846a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e606288976cfe89bc00ac47f1602a99e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 46, 70], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d89e84045749a23ea4d0eed6696846a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e606288976cfe89bc00ac47f1602a99e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 46, 70], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d89e84045749a23ea4d0eed6696846a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e606288976cfe89bc00ac47f1602a99e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 46, 70], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d89e84045749a23ea4d0eed6696846a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e606288976cfe89bc00ac47f1602a99e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 46, 70], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d89e84045749a23ea4d0eed6696846a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e606288976cfe89bc00ac47f1602a99e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 46, 70], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_87405b4dc4a86bf24d64e7fa8322e3a4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e606288976cfe89bc00ac47f1602a99e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 23, 35], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_87405b4dc4a86bf24d64e7fa8322e3a4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e606288976cfe89bc00ac47f1602a99e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 23, 35], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_87405b4dc4a86bf24d64e7fa8322e3a4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e606288976cfe89bc00ac47f1602a99e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 23, 35], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_87405b4dc4a86bf24d64e7fa8322e3a4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e606288976cfe89bc00ac47f1602a99e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 23, 35], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_87405b4dc4a86bf24d64e7fa8322e3a4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e606288976cfe89bc00ac47f1602a99e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 23, 35], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_87405b4dc4a86bf24d64e7fa8322e3a4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e606288976cfe89bc00ac47f1602a99e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 23, 35], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_87405b4dc4a86bf24d64e7fa8322e3a4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e606288976cfe89bc00ac47f1602a99e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 23, 35], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_87405b4dc4a86bf24d64e7fa8322e3a4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e606288976cfe89bc00ac47f1602a99e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 23, 35], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f254e52d466aa4171c890374089f2a4f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e606288976cfe89bc00ac47f1602a99e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 12, 18], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f254e52d466aa4171c890374089f2a4f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e606288976cfe89bc00ac47f1602a99e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 12, 18], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f254e52d466aa4171c890374089f2a4f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e606288976cfe89bc00ac47f1602a99e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 12, 18], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f254e52d466aa4171c890374089f2a4f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e606288976cfe89bc00ac47f1602a99e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 12, 18], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f254e52d466aa4171c890374089f2a4f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e606288976cfe89bc00ac47f1602a99e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 12, 18], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f254e52d466aa4171c890374089f2a4f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e606288976cfe89bc00ac47f1602a99e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 12, 18], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f254e52d466aa4171c890374089f2a4f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e606288976cfe89bc00ac47f1602a99e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 12, 18], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f254e52d466aa4171c890374089f2a4f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e606288976cfe89bc00ac47f1602a99e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 12, 18], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_aa894ce9925afa7eb52f8780831dc616(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e606288976cfe89bc00ac47f1602a99e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 6, 9], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_aa894ce9925afa7eb52f8780831dc616(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e606288976cfe89bc00ac47f1602a99e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 6, 9], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_aa894ce9925afa7eb52f8780831dc616(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e606288976cfe89bc00ac47f1602a99e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 6, 9], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_aa894ce9925afa7eb52f8780831dc616(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e606288976cfe89bc00ac47f1602a99e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 6, 9], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_aa894ce9925afa7eb52f8780831dc616(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e606288976cfe89bc00ac47f1602a99e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 6, 9], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_aa894ce9925afa7eb52f8780831dc616(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e606288976cfe89bc00ac47f1602a99e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 6, 9], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_aa894ce9925afa7eb52f8780831dc616(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e606288976cfe89bc00ac47f1602a99e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 6, 9], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_aa894ce9925afa7eb52f8780831dc616(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e606288976cfe89bc00ac47f1602a99e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 6, 9], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_27c92801d47ff9323f63973d0584b760(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b06f3f66f48b1468907b194d0315299d
    def get_inputs(self):
        return [
            paddle.uniform([1, 84, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_75d609ee63ac513d40511c773b69967a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6553b0c676e0a751f4fb6f9c2cdeec02
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f1fe298d8d1420a6d3429c6fb3abc7f7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_010c79a5cb2ec4c93bd2b08a2c53d251
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[3.4693686962127686]], [[3.907703399658203]], [[4.54768705368042]], [[4.221058368682861]], [[3.591783285140991]], [[3.7721452713012695]], [[3.972290277481079]], [[3.8032846450805664]], [[3.9062094688415527]], [[3.5290470123291016]], [[3.496199131011963]], [[2.8817920684814453]], [[3.7164390087127686]], [[3.590005874633789]]]], dtype='float32').reshape([1, 14, 1, 1]),
        ]


class TestPrimitiveOp_56a43d75681ded3a4f0de1d28d84bd8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c9d1c2dd10643e5c51aa4bd2b0fdb4c7
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[4.408426761627197]], [[4.580110549926758]], [[5.124971389770508]], [[5.724294185638428]], [[4.407931804656982]], [[5.442280292510986]], [[5.664060115814209]], [[4.495202541351318]], [[4.995047092437744]], [[4.68555212020874]], [[4.894424915313721]], [[3.941249370574951]], [[5.2427473068237305]], [[4.67960786819458]], [[5.031906604766846]], [[4.981894493103027]], [[4.324416160583496]], [[4.9982123374938965]], [[4.808460712432861]], [[5.1347432136535645]]]], dtype='float32').reshape([1, 20, 1, 1]),
        ]


class TestPrimitiveOp_744c0a907d28f724ea66782fb27258f6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e606288976cfe89bc00ac47f1602a99e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 14, 20], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f0ecb0af7de4fd68cf8293df74f74e74(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a4dcbfeac1ca8c9f4347b7a2c270ab31
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_cda531f912e3a2683811a2fde7cee6ab(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e959870eeefa9ce0c62dc82cca569d99
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[7.657050609588623]], [[7.467957019805908]], [[8.277344703674316]], [[7.793003559112549]], [[7.169888973236084]], [[7.580733299255371]], [[7.163868427276611]], [[8.383652687072754]], [[7.745558738708496]], [[7.415563106536865]], [[8.139208793640137]], [[6.923257350921631]], [[7.756600856781006]], [[7.330754280090332]], [[8.161508560180664]], [[7.202711582183838]], [[8.36557674407959]], [[7.334307670593262]], [[8.077698707580566]], [[8.276175498962402]], [[6.9406938552856445]], [[7.868319034576416]], [[7.285653114318848]], [[7.849115371704102]], [[8.189362525939941]], [[7.556140422821045]], [[7.912625789642334]], [[8.571578025817871]], [[7.268167018890381]], [[7.441878795623779]]]], dtype='float32').reshape([1, 30, 1, 1]),
        ]


class TestPrimitiveOp_92de2d354ff394a03d56f5a4bb353bef(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_596dd1ada5fe26f97607c6e853778e25
    def get_inputs(self):
        return [
            paddle.uniform([1, 240, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_280d13cfc4843d055eba8d88a90105e2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_087f1861b31113457822ac08f97c3338
    def get_inputs(self):
        return [
            paddle.uniform([1, 40, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_bf9afed14dfe99d3c5fc94eb976b9aac(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_65a7d4cc877c28cdc806b0207d0c244f
    def get_inputs(self):
        return [
            paddle.uniform([1, 36, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3dc124ca247fd1f7113ce100ef98170c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1774c9ccedef3922581dcc40dd7fe6be
    def get_inputs(self):
        return [
            paddle.uniform([22, 96, 109, 109], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6d5aab49ba4d1938ee5406d68ae72bd9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_775603205c393624833ba935d4ef7239
    def get_inputs(self):
        return [
            paddle.uniform([22, 16, 54, 54], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6797735e5763817a90215840e38d9769(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_081b0c899a155f8e8b9b92f30e16c3ce
    def get_inputs(self):
        return [
            paddle.uniform([22, 64, 54, 54], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6797735e5763817a90215840e38d9769(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_081b0c899a155f8e8b9b92f30e16c3ce
    def get_inputs(self):
        return [
            paddle.uniform([22, 64, 54, 54], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6d5aab49ba4d1938ee5406d68ae72bd9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_775603205c393624833ba935d4ef7239
    def get_inputs(self):
        return [
            paddle.uniform([22, 16, 54, 54], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6797735e5763817a90215840e38d9769(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_081b0c899a155f8e8b9b92f30e16c3ce
    def get_inputs(self):
        return [
            paddle.uniform([22, 64, 54, 54], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6797735e5763817a90215840e38d9769(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_081b0c899a155f8e8b9b92f30e16c3ce
    def get_inputs(self):
        return [
            paddle.uniform([22, 64, 54, 54], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ef46b83592fecb8f8d53eee98234875b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_017d29a1e4094c0e6532eff8c6d07cea
    def get_inputs(self):
        return [
            paddle.uniform([22, 32, 54, 54], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_5e630342bde82fe0ad7e223875e26389(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e4adcd75dc29b0fdba55384f372344a0
    def get_inputs(self):
        return [
            paddle.uniform([22, 128, 54, 54], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_5e630342bde82fe0ad7e223875e26389(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e4adcd75dc29b0fdba55384f372344a0
    def get_inputs(self):
        return [
            paddle.uniform([22, 128, 54, 54], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_1eb2aa5157109faae665a9f8bb9dd08d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_017d29a1e4094c0e6532eff8c6d07cea
    def get_inputs(self):
        return [
            paddle.uniform([22, 32, 26, 26], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_135cc4d5fd9a3457dd16271ad946d487(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e4adcd75dc29b0fdba55384f372344a0
    def get_inputs(self):
        return [
            paddle.uniform([22, 128, 26, 26], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_135cc4d5fd9a3457dd16271ad946d487(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e4adcd75dc29b0fdba55384f372344a0
    def get_inputs(self):
        return [
            paddle.uniform([22, 128, 26, 26], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_65532f9d914111af8275b20a0dd20a87(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_531f1e2922208284d24d0824995301f7
    def get_inputs(self):
        return [
            paddle.uniform([22, 48, 26, 26], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_830fd4efe859f317e3ff286d557e5589(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9e25c2bd90b4a2216ed623d19f6b43fd
    def get_inputs(self):
        return [
            paddle.uniform([22, 192, 26, 26], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_830fd4efe859f317e3ff286d557e5589(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9e25c2bd90b4a2216ed623d19f6b43fd
    def get_inputs(self):
        return [
            paddle.uniform([22, 192, 26, 26], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_65532f9d914111af8275b20a0dd20a87(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_531f1e2922208284d24d0824995301f7
    def get_inputs(self):
        return [
            paddle.uniform([22, 48, 26, 26], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_830fd4efe859f317e3ff286d557e5589(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9e25c2bd90b4a2216ed623d19f6b43fd
    def get_inputs(self):
        return [
            paddle.uniform([22, 192, 26, 26], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_830fd4efe859f317e3ff286d557e5589(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9e25c2bd90b4a2216ed623d19f6b43fd
    def get_inputs(self):
        return [
            paddle.uniform([22, 192, 26, 26], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_801cfc85fd06530ab9fe8766891b6f55(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_081b0c899a155f8e8b9b92f30e16c3ce
    def get_inputs(self):
        return [
            paddle.uniform([22, 64, 26, 26], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_36e21f56c51298172bb5f427cb08641a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e606288976cfe89bc00ac47f1602a99e
    def get_inputs(self):
        return [
            paddle.uniform([22, 256, 26, 26], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_36e21f56c51298172bb5f427cb08641a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e606288976cfe89bc00ac47f1602a99e
    def get_inputs(self):
        return [
            paddle.uniform([22, 256, 26, 26], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e5ab7b0657d90d40a432b6e8bf27b6a2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_081b0c899a155f8e8b9b92f30e16c3ce
    def get_inputs(self):
        return [
            paddle.uniform([22, 64, 12, 12], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_5d4465a61fecf22f7561e64a6ba69577(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e606288976cfe89bc00ac47f1602a99e
    def get_inputs(self):
        return [
            paddle.uniform([22, 256, 12, 12], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_5d4465a61fecf22f7561e64a6ba69577(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e606288976cfe89bc00ac47f1602a99e
    def get_inputs(self):
        return [
            paddle.uniform([22, 256, 12, 12], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ae81286bdf4153e56e2539a0332e8166(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_349a2e2d5ec0a758640b81356bde0d49
    def get_inputs(self):
        return [
            paddle.uniform([22, 1000, 12, 12], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_92d5f7991565c0ffba701eaff8d4bdef(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_99ac09e579e7234a2c35cc9891811417
    def get_inputs(self):
        return [
            paddle.uniform([1, 50, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_0aea6dd7220500f32bf019fe45cfbbdb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e606288976cfe89bc00ac47f1602a99e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 10, 16], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_05dcf9e10fa8ef983b5600250e098b8f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_79f056f364567b2465724fd6a8e83a1c
    def get_inputs(self):
        return [
            paddle.uniform([1, 100, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_40dc2219b8a13062990d710966d23851(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0a358aee5a558727aee7da4d8b921119
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[5.66949462890625]], [[5.3246588706970215]], [[5.861870288848877]], [[6.120779037475586]], [[5.858870506286621]], [[5.489810943603516]], [[6.855908393859863]], [[6.27374792098999]], [[6.702110290527344]], [[6.429392337799072]], [[5.311581611633301]], [[5.801332473754883]], [[6.015199661254883]], [[5.2471842765808105]], [[5.690392971038818]], [[5.694524765014648]], [[5.665214538574219]], [[5.505845069885254]], [[6.274290561676025]], [[5.99553108215332]], [[6.439048767089844]], [[6.097180366516113]], [[5.78026008605957]], [[5.487403392791748]]]], dtype='float32').reshape([1, 24, 1, 1]),
        ]


class TestPrimitiveOp_b447d4d18e56f5fc19082d2c974eb905(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4617feab09e563b51a144da544da4870
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[6.102778434753418]], [[6.755051612854004]], [[6.838598251342773]], [[6.176799774169922]], [[6.563821792602539]], [[6.656938076019287]], [[6.842562675476074]], [[6.7747697830200195]], [[7.549343585968018]], [[6.687005043029785]], [[7.159262657165527]], [[6.703189849853516]], [[6.13550329208374]], [[7.844245910644531]], [[6.841195106506348]], [[6.4444169998168945]], [[6.278687000274658]], [[6.766174793243408]], [[6.057496547698975]], [[6.79707670211792]], [[7.025205612182617]], [[6.469728946685791]], [[6.18354606628418]], [[6.685203552246094]], [[6.819109916687012]]]], dtype='float32').reshape([1, 25, 1, 1]),
        ]


class TestPrimitiveOp_6eb43361abeea1932b84036bfefdd510(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6bf861630a81723a2cec98b4fa32a9d3
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[3.3094823360443115]], [[3.546820878982544]], [[3.0335607528686523]], [[3.33363938331604]], [[2.4180808067321777]], [[3.32660174369812]], [[3.3581957817077637]], [[3.267660140991211]], [[3.2092204093933105]], [[3.141242504119873]], [[3.2115492820739746]], [[3.2843070030212402]]]], dtype='float32').reshape([1, 12, 1, 1]),
        ]


class TestPrimitiveOp_dc8ea1d2e983d22632e0728730a19b98(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e97529b02f651226bea345ddc53c376f
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_280d13cfc4843d055eba8d88a90105e2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_087f1861b31113457822ac08f97c3338
    def get_inputs(self):
        return [
            paddle.uniform([1, 40, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f0ecb0af7de4fd68cf8293df74f74e74(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a4dcbfeac1ca8c9f4347b7a2c270ab31
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_1e230f287cabff66142752176cf2698d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e606288976cfe89bc00ac47f1602a99e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 10, 15], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_c00de39914eaeee5d51871a73dba4a78(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 64, 25, 38], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_c386af5a19cb484af8ef35ae4e7795d7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c00de39914eaeee5d51871a73dba4a78
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 25, 38], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_737c791a89c898fc04847f4ce0c9162b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d2b6ec29c6007cab5cc9a49705a3b96e
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_42452863f6f3275dd04e72208de9dd29(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_78aa65d9647b578c2f80bf46dde41aaa
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 112, 160], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_280d13cfc4843d055eba8d88a90105e2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_087f1861b31113457822ac08f97c3338
    def get_inputs(self):
        return [
            paddle.uniform([1, 40, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_dc8ea1d2e983d22632e0728730a19b98(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e97529b02f651226bea345ddc53c376f
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_7044aea9e9fcfbc6ec01ea8b1cbcf47f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 64, 7, 10], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_d4add8260c6792a5c475dfa7b959c77b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7044aea9e9fcfbc6ec01ea8b1cbcf47f
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 7, 10], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_058cb602a9143fc6d3fe23ef2a7a32cd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0a358aee5a558727aee7da4d8b921119
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[643.7966918945312]], [[669.8527221679688]], [[650.4267578125]], [[688.0873413085938]], [[673.0556030273438]], [[700.4279174804688]], [[678.2283325195312]], [[606.1294555664062]], [[681.2627563476562]], [[665.9684448242188]], [[700.55419921875]], [[637.4541625976562]], [[723.3150634765625]], [[653.7767944335938]], [[669.7727661132812]], [[672.3577880859375]], [[658.6781616210938]], [[705.0426635742188]], [[635.1400756835938]], [[642.2073974609375]], [[742.2282104492188]], [[738.646484375]], [[631.125732421875]], [[638.9788208007812]]]], dtype='float32').reshape([1, 24, 1, 1]),
        ]


class TestPrimitiveOp_1591f37b94926503393b19a90a9f53b5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0a358aee5a558727aee7da4d8b921119
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[77.27014923095703]], [[76.04459381103516]], [[83.39509582519531]], [[80.14073944091797]], [[80.18385314941406]], [[79.04598999023438]], [[81.0997314453125]], [[83.60243225097656]], [[86.23796844482422]], [[75.66444396972656]], [[83.40579986572266]], [[83.80780792236328]], [[87.89502716064453]], [[89.23097229003906]], [[87.5143051147461]], [[85.46206665039062]], [[80.84929656982422]], [[89.4579849243164]], [[94.40455627441406]], [[84.7237319946289]], [[91.21224212646484]], [[83.22432708740234]], [[84.1407241821289]], [[74.48222351074219]]]], dtype='float32').reshape([1, 24, 1, 1]),
        ]


class TestPrimitiveOp_d13f27e4a9eb2dd6e5cdd80122ac59a2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0a358aee5a558727aee7da4d8b921119
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[23.455074310302734]], [[22.37508201599121]], [[22.317672729492188]], [[23.183090209960938]], [[24.32966423034668]], [[22.778457641601562]], [[23.396730422973633]], [[23.23088836669922]], [[23.24786376953125]], [[22.884109497070312]], [[23.53722381591797]], [[23.473379135131836]], [[24.2813777923584]], [[21.681015014648438]], [[23.83696174621582]], [[21.554962158203125]], [[23.11602210998535]], [[23.789596557617188]], [[22.00689125061035]], [[20.594526290893555]], [[23.3435001373291]], [[25.405641555786133]], [[23.160402297973633]], [[21.694786071777344]]]], dtype='float32').reshape([1, 24, 1, 1]),
        ]


class TestPrimitiveOp_0fa78645d8527e677347770b4323131a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0a358aee5a558727aee7da4d8b921119
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[17.765321731567383]], [[19.559804916381836]], [[19.31717300415039]], [[18.151464462280273]], [[18.402856826782227]], [[18.293052673339844]], [[18.870729446411133]], [[18.670217514038086]], [[17.8748722076416]], [[18.503948211669922]], [[18.10445213317871]], [[19.237512588500977]], [[18.114986419677734]], [[17.788070678710938]], [[20.259672164916992]], [[16.1656551361084]], [[17.78622817993164]], [[19.533567428588867]], [[19.575441360473633]], [[19.39875030517578]], [[18.0234375]], [[17.64359474182129]], [[17.5723876953125]], [[18.47214698791504]]]], dtype='float32').reshape([1, 24, 1, 1]),
        ]


class TestPrimitiveOp_47aab5490d0c8a2d5813e27eb19f062a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_99313ba4af9dea5fe586f9a06d4e7c29
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[30620.724609375]], [[32545.693359375]], [[25545.333984375]], [[32017.056640625]], [[38487.1875]], [[35382.5859375]]]], dtype='float32').reshape([1, 6, 1, 1]),
        ]


class TestPrimitiveOp_02c1885bc2b5cf1012219be2141e30c6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_99313ba4af9dea5fe586f9a06d4e7c29
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[30364.189453125]], [[39599.1328125]], [[33627.359375]], [[36777.046875]], [[49147.08984375]], [[43103.75]]]], dtype='float32').reshape([1, 6, 1, 1]),
        ]


class TestPrimitiveOp_3461c6ee52ad2de930f528bba4214efb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_99313ba4af9dea5fe586f9a06d4e7c29
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[41908.5]], [[36754.4765625]], [[40769.3203125]], [[41982.7734375]], [[34369.89453125]], [[35136.10546875]]]], dtype='float32').reshape([1, 6, 1, 1]),
        ]


class TestPrimitiveOp_764d898ab73fc3896cb33aa5645632ae(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_99313ba4af9dea5fe586f9a06d4e7c29
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[41434.21875]], [[38502.94921875]], [[38644.33203125]], [[33797.2578125]], [[36343.7734375]], [[37715.55859375]]]], dtype='float32').reshape([1, 6, 1, 1]),
        ]


class TestPrimitiveOp_e6bec40b5ac2de32ac45f07215e3d527(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_78aa65d9647b578c2f80bf46dde41aaa
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 11, 17], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_92de2d354ff394a03d56f5a4bb353bef(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_596dd1ada5fe26f97607c6e853778e25
    def get_inputs(self):
        return [
            paddle.uniform([1, 240, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d62e6d97418e6c563a1730b131af3b54(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_94fa2d9f0e4164dc0e384b4c316906cc
    def get_inputs(self):
        return [
            paddle.uniform([1, 72, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_1f8819dde398899c04f539dad109fa83(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_78aa65d9647b578c2f80bf46dde41aaa
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 88, 132], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_bb6bed69ca167aa2ba13e6e8ca6f81e0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0a358aee5a558727aee7da4d8b921119
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[6.7747578620910645]], [[7.277425289154053]], [[7.628300189971924]], [[6.7336578369140625]], [[6.166509628295898]], [[6.037654399871826]], [[6.47376823425293]], [[6.21839714050293]], [[6.23767614364624]], [[6.518399715423584]], [[6.108009338378906]], [[6.963770389556885]], [[6.03821325302124]], [[7.372227668762207]], [[7.316105365753174]], [[6.159783363342285]], [[7.72247838973999]], [[6.894965171813965]], [[6.657410144805908]], [[6.8110432624816895]], [[6.489583492279053]], [[6.812068939208984]], [[6.552923202514648]], [[6.3407721519470215]]]], dtype='float32').reshape([1, 24, 1, 1]),
        ]


class PrimitiveOp_3c6e0cb370fa66c2ee31c49aac1b354f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 64, 100, 152], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_bbd350de28f20ed3693df24f82d6ebf3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3c6e0cb370fa66c2ee31c49aac1b354f
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 100, 152], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_5152ae2f4acc0de14ca78c19ce8e8d64(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 156], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_b2f3f762a14a40bf962b39dca08a0731(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5152ae2f4acc0de14ca78c19ce8e8d64
    def get_inputs(self):
        return [
            paddle.uniform([1, 156], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_2a572e355636ffe6b28ad3970ede1011(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fb377a9795497c0bdea579bf8eeb09fe
    def get_inputs(self):
        return [
            paddle.uniform([1, 168, 1, 1], dtype='float32', min=0, max=0.5),
        ]




if __name__ == '__main__':
    unittest.main()