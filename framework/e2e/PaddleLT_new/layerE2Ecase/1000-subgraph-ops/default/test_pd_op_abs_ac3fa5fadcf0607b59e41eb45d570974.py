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


class TestPrimitiveOp_afa9cb3dc0c33a8e4349dfe210f7bb60(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0744b88908a158c9736c500ad1d8b491
    def get_inputs(self):
        return [
            paddle.uniform([1841, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_2feda293e38a5aa5d86cb4e5990c346f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_44ae888f422ed4f4229b2aa25a9ec133
    def get_inputs(self):
        return [
            paddle.to_tensor([[-0.06600350141525269, 0.12325924634933472, 0.14050322771072388, -0.09338384866714478], [0.2529301643371582, 0.19529643654823303, -0.035754598677158356, 0.3732248842716217], [0.05762083828449249, -0.33048099279403687, -0.3098161816596985, -0.046964749693870544], [-0.1361105740070343, 0.39801257848739624, 0.07723602652549744, 0.10689233243465424], [0.21829457581043243, 0.25948721170425415, 0.19501274824142456, 0.08205263316631317]], dtype='float32').reshape([5, 4]),
        ]


class TestPrimitiveOp_f5f0f6001c4e0d9e390affedb4e7f200(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_44ae888f422ed4f4229b2aa25a9ec133
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.29332220554351807, -0.17943236231803894, 0.0017442405223846436, 0.10830046236515045], [-0.04921819269657135, -0.21070674061775208, 0.37740403413772583, 0.13908714056015015], [0.02563472092151642, 0.16697435081005096, -0.006776377558708191, -0.11909252405166626], [-0.04921819269657135, -0.21070674061775208, 0.37740403413772583, 0.13908714056015015], [0.02563472092151642, 0.16697435081005096, -0.006776377558708191, -0.11909252405166626]], dtype='float32').reshape([5, 4]),
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


class TestPrimitiveOp_8d957a6070f072f15bdf73af33329332(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0744b88908a158c9736c500ad1d8b491
    def get_inputs(self):
        return [
            paddle.uniform([5562, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_5ee691ecfcd4e977f7e7fcc4b36f5c7b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_44ae888f422ed4f4229b2aa25a9ec133
    def get_inputs(self):
        return [
            paddle.to_tensor([[-0.42621099948883057, 0.1255248486995697, 0.13389798998832703, 0.25021958351135254], [0.09774552285671234, -0.2969127297401428, -0.33496904373168945, 0.39779192209243774], [0.06356087327003479, 0.019604787230491638, 0.08770968019962311, 0.3853555917739868], [0.09774552285671234, -0.2969127297401428, -0.33496904373168945, 0.39779192209243774], [0.06356087327003479, 0.019604787230491638, 0.08770968019962311, 0.3853555917739868], [-0.10860182344913483, -0.15528208017349243, 0.3284634053707123, -0.05508685111999512], [-0.10860182344913483, -0.15528208017349243, 0.3284634053707123, -0.05508685111999512]], dtype='float32').reshape([7, 4]),
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


class TestPrimitiveOp_9d39d9fd1063622df24376947a5eb2a7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0744b88908a158c9736c500ad1d8b491
    def get_inputs(self):
        return [
            paddle.uniform([1734, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3ec9e7dda1b880fcc67f6ed6628feffe(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0744b88908a158c9736c500ad1d8b491
    def get_inputs(self):
        return [
            paddle.uniform([1541, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6d8bfdc51f349923ac5c7401690be3f3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_44ae888f422ed4f4229b2aa25a9ec133
    def get_inputs(self):
        return [
            paddle.to_tensor([[-0.0347633957862854, 0.2688142955303192, 0.15619760751724243, -0.12203063070774078], [-0.04207935929298401, 0.25613322854042053, -0.023173600435256958, 0.18292436003684998], [0.06694593280553818, 0.13091182708740234, -0.36519718170166016, 0.13943824172019958], [0.054342061281204224, -0.1302337646484375, -0.2963424324989319, 0.08765411376953125], [0.054342061281204224, -0.1302337646484375, -0.2963424324989319, 0.08765411376953125], [0.06694593280553818, 0.13091182708740234, -0.36519718170166016, 0.13943824172019958]], dtype='float32').reshape([6, 4]),
        ]


class TestPrimitiveOp_e40de2b758090b9976a573fa34405d12(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_44ae888f422ed4f4229b2aa25a9ec133
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.16947636008262634, 0.12996846437454224, 0.0761517733335495, -0.06142064929008484], [-0.010005325078964233, 0.010751783847808838, 0.16052129864692688, -0.20979206264019012], [-0.3373373746871948, -0.017422378063201904, -0.02375468611717224, -0.03213486075401306], [-0.3443434238433838, 0.40481501817703247, -0.048192352056503296, -0.11602441966533661], [0.16947636008262634, 0.12996846437454224, 0.0761517733335495, -0.06142064929008484]], dtype='float32').reshape([5, 4]),
        ]


class TestPrimitiveOp_f7ee127404489ac713dd64b92089fd22(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_44ae888f422ed4f4229b2aa25a9ec133
    def get_inputs(self):
        return [
            paddle.uniform([10, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d458428c3add89034b8805c468cc3a2c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_44ae888f422ed4f4229b2aa25a9ec133
    def get_inputs(self):
        return [
            paddle.to_tensor([[-0.12086917459964752, -0.08581306040287018, -0.04574468731880188, -0.38157597184181213], [-0.4106341600418091, -0.02336856722831726, 0.06159588694572449, -0.03985142707824707], [-0.2122405618429184, -0.05072645843029022, 0.11940930038690567, -0.0011862218379974365], [-0.2085941582918167, -0.21888814866542816, 0.11163263022899628, 0.31563934683799744]], dtype='float32').reshape([4, 4]),
        ]


class TestPrimitiveOp_748a0fcd9c8c1b0162066c1dc314b930(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_44ae888f422ed4f4229b2aa25a9ec133
    def get_inputs(self):
        return [
            paddle.uniform([84, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6a22ca5021592043198648c893f83c70(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0744b88908a158c9736c500ad1d8b491
    def get_inputs(self):
        return [
            paddle.uniform([2061, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_cb5a70ae444165eaaba5675d6649fba7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_44ae888f422ed4f4229b2aa25a9ec133
    def get_inputs(self):
        return [
            paddle.to_tensor([[-0.06793880462646484, 0.1796111911535263, -0.41419026255607605, 0.23429150879383087], [-0.06793880462646484, 0.1796111911535263, -0.41419026255607605, 0.23429150879383087], [0.20293423533439636, -0.042559683322906494, -0.03312063217163086, 0.017667576670646667], [-0.39120301604270935, -0.03000320866703987, -0.09842769801616669, 0.34845295548439026], [-0.12772080302238464, -0.18704058229923248, -0.3924790620803833, 0.21641282737255096], [0.36120250821113586, -0.09090563654899597, 0.06719005107879639, 0.23598814010620117], [0.01706467568874359, -0.13718575239181519, -0.23035810887813568, -0.017347488552331924]], dtype='float32').reshape([7, 4]),
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


class TestPrimitiveOp_3ed649b7a9f65edf094a646341286cf4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0744b88908a158c9736c500ad1d8b491
    def get_inputs(self):
        return [
            paddle.uniform([4642, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3835139d1e68fbe7bddd0dedd529f99f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0744b88908a158c9736c500ad1d8b491
    def get_inputs(self):
        return [
            paddle.uniform([1042, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_bafe2f22e998d5851940138da3eb934a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_44ae888f422ed4f4229b2aa25a9ec133
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.0011395514011383057, 0.3579179644584656, 0.19031499326229095, 0.3789938688278198], [-0.14381445944309235, 0.26047101616859436, -0.28549501299858093, -0.2758251428604126], [-0.14381445944309235, 0.26047101616859436, -0.28549501299858093, -0.2758251428604126], [0.1933608502149582, 0.1633141189813614, -0.14810127019882202, -0.1474897861480713], [0.06986059248447418, -0.08585286140441895, 0.023728124797344208, 0.2839673161506653], [0.02392372488975525, -0.06696397066116333, -0.20791184902191162, 0.3662828803062439]], dtype='float32').reshape([6, 4]),
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


class TestPrimitiveOp_3d1e3cda8f51fa55ec4d4451ca666ba2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0744b88908a158c9736c500ad1d8b491
    def get_inputs(self):
        return [
            paddle.uniform([2369, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a0e459e1033d6bab58ea0debfc67a9c3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0744b88908a158c9736c500ad1d8b491
    def get_inputs(self):
        return [
            paddle.uniform([3054, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b9c318e027d0e6f4db7e6f048d5740cf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0744b88908a158c9736c500ad1d8b491
    def get_inputs(self):
        return [
            paddle.uniform([3819, 4], dtype='float32', min=0, max=0.5),
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


class TestPrimitiveOp_30d06846b038f0fb38f5d1e97f51346e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0744b88908a158c9736c500ad1d8b491
    def get_inputs(self):
        return [
            paddle.uniform([2092, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_5b554c694ee987feeaa3318114089415(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_935aa3778f50ab3eef1173d72ce6f082
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 32, 32], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c9d299ad764bab76fe0fada3a752fa33(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_44ae888f422ed4f4229b2aa25a9ec133
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.22915200889110565, -0.006040988024324179, 0.13718536496162415, -0.06582275778055191], [0.00850994884967804, 0.03199875354766846, 0.10096690058708191, 0.31683748960494995], [-0.03647342324256897, -0.18587401509284973, -0.20138487219810486, -0.0098896324634552], [-0.03647342324256897, -0.18587401509284973, -0.20138487219810486, -0.0098896324634552], [0.025345072150230408, 0.1740993857383728, -0.031156376004219055, 0.023688942193984985]], dtype='float32').reshape([5, 4]),
        ]


class TestPrimitiveOp_4eee3c2f51c339b160066afba57fa6db(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_44ae888f422ed4f4229b2aa25a9ec133
    def get_inputs(self):
        return [
            paddle.uniform([56, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a9c3abfe62f009dae98692b38a3c9d06(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0744b88908a158c9736c500ad1d8b491
    def get_inputs(self):
        return [
            paddle.uniform([4214, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_51d9e6a46fce86c8ad2ddd042dcccee0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_44ae888f422ed4f4229b2aa25a9ec133
    def get_inputs(self):
        return [
            paddle.to_tensor([[-0.17515462636947632, 0.22402986884117126, -0.05804827809333801, -0.09627231955528259], [-0.40905141830444336, -0.17959129810333252, 0.16124853491783142, 0.036101073026657104], [0.029448404908180237, 0.06972748041152954, 0.19806984066963196, 0.2812197506427765], [-0.17515462636947632, 0.22402986884117126, -0.05804827809333801, -0.09627231955528259], [0.11691299825906754, -0.40774694085121155, -0.08399210125207901, -0.022679775953292847], [-0.03554117679595947, -0.0161103755235672, 0.1517069935798645, -0.21505869925022125], [0.11691299825906754, -0.40774694085121155, -0.08399210125207901, -0.022679775953292847]], dtype='float32').reshape([7, 4]),
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