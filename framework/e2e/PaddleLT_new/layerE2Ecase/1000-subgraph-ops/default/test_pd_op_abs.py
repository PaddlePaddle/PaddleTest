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



class PrimitiveOp_44ae888f422ed4f4229b2aa25a9ec133(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.abs(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_64e4ad1ca2c204dfb75aec7dbd197b12(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_44ae888f422ed4f4229b2aa25a9ec133
    def get_inputs(self):
        return [
            paddle.uniform([1024, 5], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_fc65c0fa4979444ced9ed3e246e82a19(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_44ae888f422ed4f4229b2aa25a9ec133
    def get_inputs(self):
        return [
            paddle.uniform([4096, 5], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_41ad159df9f5863a4df392e068a1c331(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_44ae888f422ed4f4229b2aa25a9ec133
    def get_inputs(self):
        return [
            paddle.uniform([8, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d24778539ddd73a4b1153a971b438c07(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_44ae888f422ed4f4229b2aa25a9ec133
    def get_inputs(self):
        return [
            paddle.uniform([53, 4], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_0744b88908a158c9736c500ad1d8b491(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.abs(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 4], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_7405d99a051fc4c8dc6b073ef471d693(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0744b88908a158c9736c500ad1d8b491
    def get_inputs(self):
        return [
            paddle.uniform([1756, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_021761473968f65e59188a171edc4afd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_44ae888f422ed4f4229b2aa25a9ec133
    def get_inputs(self):
        return [
            paddle.to_tensor([[-0.1897001415491104, 0.22146207094192505, -0.024362623691558838, -0.2110663652420044], [-0.3173893690109253, -0.12924069166183472, 0.05257509648799896, -0.03560730814933777], [0.16317197680473328, -0.15494099259376526, 0.13420870900154114, -0.2878504693508148], [-0.13793113827705383, 0.05281302332878113, 0.19631893932819366, -0.22591465711593628], [-0.17999574542045593, 0.270733118057251, 0.054583169519901276, -0.08161468803882599]], dtype='float32').reshape([5, 4]),
        ]


class TestPrimitiveOp_1c77e444d0b3b5f7fb19aa8b71a22772(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_44ae888f422ed4f4229b2aa25a9ec133
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.14203521609306335, 0.2730371356010437, -0.009997613728046417, 0.14050546288490295], [0.2395579069852829, -0.19840773940086365, 0.1306590437889099, -0.010557323694229126], [0.1869293451309204, 0.0038643181324005127, 0.09275287389755249, -0.0023336708545684814], [0.2395579069852829, -0.19840773940086365, 0.1306590437889099, -0.010557323694229126], [0.1869293451309204, 0.0038643181324005127, 0.09275287389755249, -0.0023336708545684814]], dtype='float32').reshape([5, 4]),
        ]


class PrimitiveOp_935aa3778f50ab3eef1173d72ce6f082(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.abs(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 1, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_6e8b2c37dafafbed386a345b30aac439(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_935aa3778f50ab3eef1173d72ce6f082
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 8, 8], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a0ac65adcbf6a3a5cc0a04e90ef08469(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0744b88908a158c9736c500ad1d8b491
    def get_inputs(self):
        return [
            paddle.uniform([5551, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_53d154a92e6e5c8593efe01a85612151(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_44ae888f422ed4f4229b2aa25a9ec133
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.20235006511211395, 0.045952826738357544, 0.22046057879924774, 0.07091942429542542], [0.030943863093852997, -0.1570379137992859, 0.17190364003181458, -0.16179874539375305], [0.29283690452575684, -0.026572100818157196, 0.016743332147598267, 0.12053439021110535], [0.030943863093852997, -0.1570379137992859, 0.17190364003181458, -0.16179874539375305], [0.29283690452575684, -0.026572100818157196, 0.016743332147598267, 0.12053439021110535], [-0.2287510633468628, 0.19231140613555908, 0.0006733033806085587, -0.021374017000198364], [-0.2287510633468628, 0.19231140613555908, 0.0006733033806085587, -0.021374017000198364]], dtype='float32').reshape([7, 4]),
        ]


class TestPrimitiveOp_e04c382fce43d0eb85d7257c6c5e44e6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_44ae888f422ed4f4229b2aa25a9ec133
    def get_inputs(self):
        return [
            paddle.uniform([64, 5], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_483f8833f8eb556d809a6293b68d4fbe(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_44ae888f422ed4f4229b2aa25a9ec133
    def get_inputs(self):
        return [
            paddle.uniform([103, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_2ae130a9e3ebcecbc895bf51ee7b6539(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0744b88908a158c9736c500ad1d8b491
    def get_inputs(self):
        return [
            paddle.uniform([1769, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_49bb023af89ffae611f7fffb8920af08(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0744b88908a158c9736c500ad1d8b491
    def get_inputs(self):
        return [
            paddle.uniform([1502, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9ae4a3d05e37ca1e68613af401257037(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_44ae888f422ed4f4229b2aa25a9ec133
    def get_inputs(self):
        return [
            paddle.to_tensor([[-0.2273627370595932, -0.36241406202316284, -0.005276113748550415, -0.2186964452266693], [0.018216833472251892, 0.14161889255046844, 0.17217203974723816, 0.026601048186421394], [-0.01650775969028473, -0.12234698235988617, -0.11415546387434006, -0.17477966845035553], [-0.32719725370407104, 0.11412149667739868, -0.08081448078155518, -0.02189537324011326], [-0.32719725370407104, 0.11412149667739868, -0.08081448078155518, -0.02189537324011326], [-0.01650775969028473, -0.12234698235988617, -0.11415546387434006, -0.17477966845035553]], dtype='float32').reshape([6, 4]),
        ]


class TestPrimitiveOp_193a65113ff8a2b5ae6225c422e787b0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_44ae888f422ed4f4229b2aa25a9ec133
    def get_inputs(self):
        return [
            paddle.to_tensor([[-0.27225011587142944, 0.14648637175559998, -0.2816023528575897, -0.23074118793010712], [-0.039505332708358765, 0.376843124628067, -0.0734100341796875, 0.04012419655919075], [0.2798236608505249, 0.03169974684715271, 0.0014654099941253662, -0.14718544483184814], [-0.11550545692443848, -0.08491256833076477, -0.2013431191444397, -0.34641632437705994], [-0.27225011587142944, 0.14648637175559998, -0.2816023528575897, -0.23074118793010712]], dtype='float32').reshape([5, 4]),
        ]


class TestPrimitiveOp_f7ee127404489ac713dd64b92089fd22(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_44ae888f422ed4f4229b2aa25a9ec133
    def get_inputs(self):
        return [
            paddle.uniform([10, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_97b94fca81847168edc0cb6477b25a70(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_44ae888f422ed4f4229b2aa25a9ec133
    def get_inputs(self):
        return [
            paddle.to_tensor([[-0.2522020936012268, -0.030356958508491516, -0.0375291183590889, -0.1584610939025879], [-0.4743068814277649, 0.10842088609933853, -0.19645550847053528, 0.3040626347064972], [-0.23538586497306824, 0.18390172719955444, 0.13406015932559967, -0.012758731842041016], [-0.11915967613458633, 0.3680647611618042, 0.3414975106716156, 0.12178853899240494]], dtype='float32').reshape([4, 4]),
        ]


class TestPrimitiveOp_748a0fcd9c8c1b0162066c1dc314b930(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_44ae888f422ed4f4229b2aa25a9ec133
    def get_inputs(self):
        return [
            paddle.uniform([84, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_0ab1a364c3af2ee67f9aee09b781f2b2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0744b88908a158c9736c500ad1d8b491
    def get_inputs(self):
        return [
            paddle.uniform([2080, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_fa608a1c58ef235fa3deed50510fdcb4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_44ae888f422ed4f4229b2aa25a9ec133
    def get_inputs(self):
        return [
            paddle.to_tensor([[-0.2144443690776825, -0.07956269383430481, -0.07766185700893402, 0.23967543244361877], [-0.2144443690776825, -0.07956269383430481, -0.07766185700893402, 0.23967543244361877], [-0.010381340980529785, -0.21422593295574188, 0.07575803995132446, -0.057327091693878174], [0.06550672650337219, 0.21301709115505219, -0.1708848923444748, -0.10903717577457428], [-0.14020246267318726, -0.05354096740484238, 0.0539340078830719, 0.06818994879722595], [0.20319201052188873, 0.3426782488822937, -0.3060799241065979, -0.16592571139335632], [0.21563895046710968, -0.18433766067028046, -0.22920235991477966, 0.007676184177398682]], dtype='float32').reshape([7, 4]),
        ]


class TestPrimitiveOp_b1528af25a99d7463a71cc62d868ad58(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_44ae888f422ed4f4229b2aa25a9ec133
    def get_inputs(self):
        return [
            paddle.uniform([16384, 5], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_7c4652e777bcec7eee8e0b1d1b99070c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_935aa3778f50ab3eef1173d72ce6f082
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 64, 64], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c05c0b9edd497058f8bb32690077f4fb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0744b88908a158c9736c500ad1d8b491
    def get_inputs(self):
        return [
            paddle.uniform([4585, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_914bf6d6a351dce00fd23dc3c48ac884(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0744b88908a158c9736c500ad1d8b491
    def get_inputs(self):
        return [
            paddle.uniform([1048, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_dc903376ab9402019a533499a514b656(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_44ae888f422ed4f4229b2aa25a9ec133
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.24620190262794495, -0.07546661794185638, 0.44069913029670715, 0.30480489134788513], [0.05184207856655121, 0.3793771266937256, 0.27201610803604126, -0.27355000376701355], [0.05184207856655121, 0.3793771266937256, 0.27201610803604126, -0.27355000376701355], [0.06116703152656555, -0.007416635751724243, -0.17603977024555206, -0.288661390542984], [0.451762855052948, 0.08274058252573013, -0.015876702964305878, 0.04930630326271057], [-0.009141981601715088, 0.29538965225219727, 0.08718730509281158, 0.13560664653778076]], dtype='float32').reshape([6, 4]),
        ]


class PrimitiveOp_66ae1e3015c2ba56efe3bb5ce774f9c6(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.abs(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[100, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_0bd7e58acbe668d3859c0518f1eb6e8e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_66ae1e3015c2ba56efe3bb5ce774f9c6
    def get_inputs(self):
        return [
            paddle.uniform([100, 2, 4], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_8869bdadf09187910b8a09d7415efe29(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.abs(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[300, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_65c40af6cf1a0deced9dbf9af6ee06c8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8869bdadf09187910b8a09d7415efe29
    def get_inputs(self):
        return [
            paddle.uniform([300, 2, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_edbdcdf3e51c2601550916a43f1de581(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_935aa3778f50ab3eef1173d72ce6f082
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 128, 128], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_5930daa0d8dde35d7935667dacd00e29(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0744b88908a158c9736c500ad1d8b491
    def get_inputs(self):
        return [
            paddle.uniform([2390, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a46cbed9f9a67958963d1f3a0088d251(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0744b88908a158c9736c500ad1d8b491
    def get_inputs(self):
        return [
            paddle.uniform([3090, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ed7fabe159a2543f536d312e2ebe6654(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0744b88908a158c9736c500ad1d8b491
    def get_inputs(self):
        return [
            paddle.uniform([3748, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_7306c4069455e24e9fb2058d547993af(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_935aa3778f50ab3eef1173d72ce6f082
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 16, 16], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ae00df756f3a799b9031c12bc1dab19b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_44ae888f422ed4f4229b2aa25a9ec133
    def get_inputs(self):
        return [
            paddle.uniform([256, 5], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_403faa63566dbd8dd788bacb42a139b1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_44ae888f422ed4f4229b2aa25a9ec133
    def get_inputs(self):
        return [
            paddle.uniform([47, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_92484a18a598af997c19de24dc372fc5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0744b88908a158c9736c500ad1d8b491
    def get_inputs(self):
        return [
            paddle.uniform([2031, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_5b554c694ee987feeaa3318114089415(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_935aa3778f50ab3eef1173d72ce6f082
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 32, 32], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e47121a5eff63f15f25daa3a254885c1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_44ae888f422ed4f4229b2aa25a9ec133
    def get_inputs(self):
        return [
            paddle.to_tensor([[-0.031348638236522675, -0.033763885498046875, 0.10465838015079498, 0.04310715198516846], [0.2683018445968628, 0.28695592284202576, 0.09896203875541687, -0.3688707947731018], [0.222511425614357, 0.35377877950668335, 0.18001191318035126, -0.2595044672489166], [0.222511425614357, 0.35377877950668335, 0.18001191318035126, -0.2595044672489166], [-0.14614498615264893, -0.2770560383796692, 0.16415265202522278, -0.3097259998321533]], dtype='float32').reshape([5, 4]),
        ]


class TestPrimitiveOp_4eee3c2f51c339b160066afba57fa6db(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_44ae888f422ed4f4229b2aa25a9ec133
    def get_inputs(self):
        return [
            paddle.uniform([56, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_880dc5d8de3f40605d5e366219b90693(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0744b88908a158c9736c500ad1d8b491
    def get_inputs(self):
        return [
            paddle.uniform([4205, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b7d9bc9f8e22bf944dbac13876eaa40c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_44ae888f422ed4f4229b2aa25a9ec133
    def get_inputs(self):
        return [
            paddle.to_tensor([[-0.3006541132926941, -0.1925070732831955, -0.18089912831783295, -0.06294262409210205], [-0.07820473611354828, -0.31322187185287476, -0.34102195501327515, -0.3568349778652191], [-0.324296772480011, -0.09075477719306946, -0.10238906741142273, 0.07518288493156433], [-0.3006541132926941, -0.1925070732831955, -0.18089912831783295, -0.06294262409210205], [0.02252715826034546, -0.14715853333473206, -0.1897473931312561, 0.4024507999420166], [0.13983139395713806, -0.06232455372810364, 0.33095210790634155, -0.10818608105182648], [0.02252715826034546, -0.14715853333473206, -0.1897473931312561, 0.4024507999420166]], dtype='float32').reshape([7, 4]),
        ]


class TestPrimitiveOp_bdc626495cbe6aca7aeb9a1798e16282(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_44ae888f422ed4f4229b2aa25a9ec133
    def get_inputs(self):
        return [
            paddle.uniform([52, 4], dtype='float32', min=0, max=0.5),
        ]




if __name__ == '__main__':
    unittest.main()