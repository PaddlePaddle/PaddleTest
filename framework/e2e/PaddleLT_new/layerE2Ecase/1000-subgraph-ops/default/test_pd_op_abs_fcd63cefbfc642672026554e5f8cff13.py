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


class TestPrimitiveOp_f4825b00dfc2b91c087f428d05b108d3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0744b88908a158c9736c500ad1d8b491
    def get_inputs(self):
        return [
            paddle.uniform([1799, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_1a82d2c115162a5f5887dea9a2bac256(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_44ae888f422ed4f4229b2aa25a9ec133
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.4918801784515381, 0.2867724299430847, -0.0616503544151783, 0.29163727164268494], [0.16717244684696198, -0.2702065706253052, 0.22598737478256226, -0.018749460577964783], [0.3956667482852936, 0.03867020085453987, 0.059548020362854004, -0.3594655990600586], [-0.32695555686950684, -0.20531868934631348, -0.20880255103111267, 0.2319372296333313], [0.1041281595826149, 0.2930876910686493, -0.3555704355239868, 0.18245330452919006]], dtype='float32').reshape([5, 4]),
        ]


class TestPrimitiveOp_42783862364529379c5ac550b89caccd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_44ae888f422ed4f4229b2aa25a9ec133
    def get_inputs(self):
        return [
            paddle.to_tensor([[-0.2883334457874298, -0.2907085716724396, 0.23830997943878174, -0.09005986899137497], [0.0063690803945064545, -0.2998930811882019, -0.4675793945789337, -0.4290544390678406], [0.3478759825229645, 0.3107743263244629, -0.18506208062171936, 0.1849118173122406], [0.0063690803945064545, -0.2998930811882019, -0.4675793945789337, -0.4290544390678406], [0.3478759825229645, 0.3107743263244629, -0.18506208062171936, 0.1849118173122406]], dtype='float32').reshape([5, 4]),
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


class TestPrimitiveOp_aca9b75e3e12759b9e3ce08f0b5705f5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0744b88908a158c9736c500ad1d8b491
    def get_inputs(self):
        return [
            paddle.uniform([5504, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b6e05452f37f8d79de980327b0fdecfe(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_44ae888f422ed4f4229b2aa25a9ec133
    def get_inputs(self):
        return [
            paddle.to_tensor([[-0.1357516497373581, -0.40738099813461304, -0.44392189383506775, 0.09353798627853394], [0.03238692879676819, 0.22002668678760529, -0.4370597302913666, -0.05804285407066345], [-0.22183875739574432, -0.4310569763183594, 0.3100452423095703, 0.13087481260299683], [0.03238692879676819, 0.22002668678760529, -0.4370597302913666, -0.05804285407066345], [-0.22183875739574432, -0.4310569763183594, 0.3100452423095703, 0.13087481260299683], [-0.20530103147029877, -0.26093873381614685, 0.25597333908081055, -0.21422967314720154], [-0.20530103147029877, -0.26093873381614685, 0.25597333908081055, -0.21422967314720154]], dtype='float32').reshape([7, 4]),
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


class TestPrimitiveOp_f2fbe2c33b64491b259e2f60b6476201(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0744b88908a158c9736c500ad1d8b491
    def get_inputs(self):
        return [
            paddle.uniform([1811, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4ad5bf8cbb20bf2fe102b89af20aec54(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0744b88908a158c9736c500ad1d8b491
    def get_inputs(self):
        return [
            paddle.uniform([1559, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_54f109811af11d42c861be1e85ab88bc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_44ae888f422ed4f4229b2aa25a9ec133
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.31275293231010437, -0.14918148517608643, 0.39206165075302124, 0.28137341141700745], [-0.0677771344780922, 0.31497669219970703, -0.3975636065006256, 0.2412356436252594], [-0.005561307072639465, -0.018071070313453674, -0.16048210859298706, 0.16573911905288696], [-0.041177887469530106, 0.02563190460205078, -0.13773715496063232, 0.11778953671455383], [-0.041177887469530106, 0.02563190460205078, -0.13773715496063232, 0.11778953671455383], [-0.005561307072639465, -0.018071070313453674, -0.16048210859298706, 0.16573911905288696]], dtype='float32').reshape([6, 4]),
        ]


class TestPrimitiveOp_c433dff12182c9ccda483e706d148e85(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_44ae888f422ed4f4229b2aa25a9ec133
    def get_inputs(self):
        return [
            paddle.to_tensor([[-0.22132810950279236, 0.2636357545852661, -0.0795370489358902, -0.10273897647857666], [-0.1800602227449417, -0.3080964982509613, 0.04047614336013794, 0.03155049681663513], [0.11648650467395782, 0.05020785331726074, -0.03515303134918213, 0.061091840267181396], [-0.01956409215927124, -0.46025997400283813, 0.06443309783935547, -0.19794350862503052], [-0.22132810950279236, 0.2636357545852661, -0.0795370489358902, -0.10273897647857666]], dtype='float32').reshape([5, 4]),
        ]


class TestPrimitiveOp_f7ee127404489ac713dd64b92089fd22(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_44ae888f422ed4f4229b2aa25a9ec133
    def get_inputs(self):
        return [
            paddle.uniform([10, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d7b09e0c05318c543be4d059bbea1291(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_44ae888f422ed4f4229b2aa25a9ec133
    def get_inputs(self):
        return [
            paddle.to_tensor([[-0.23681814968585968, -0.03353661298751831, -0.016777783632278442, -0.28043243288993835], [0.04496394097805023, 0.09608559310436249, 0.12442414462566376, -0.19400113821029663], [-0.04946696758270264, -0.23147156834602356, 0.2051411271095276, 0.018683740869164467], [0.058955393731594086, 0.19874268770217896, 0.1565515100955963, -0.160660982131958]], dtype='float32').reshape([4, 4]),
        ]


class TestPrimitiveOp_748a0fcd9c8c1b0162066c1dc314b930(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_44ae888f422ed4f4229b2aa25a9ec133
    def get_inputs(self):
        return [
            paddle.uniform([84, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a801f292e3947b23afb467bbf4eb0fe6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0744b88908a158c9736c500ad1d8b491
    def get_inputs(self):
        return [
            paddle.uniform([2066, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_af4c466039be602ad666ee3b16032c76(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_44ae888f422ed4f4229b2aa25a9ec133
    def get_inputs(self):
        return [
            paddle.to_tensor([[-0.05286979675292969, 0.135944664478302, -0.20292210578918457, 0.25536227226257324], [-0.05286979675292969, 0.135944664478302, -0.20292210578918457, 0.25536227226257324], [0.1636086106300354, 0.02989518642425537, -0.12079188227653503, 0.2221795618534088], [0.18770182132720947, 0.15640157461166382, -0.0009250938892364502, 0.22462987899780273], [0.1361793726682663, 0.13610488176345825, 0.10240694880485535, -0.13521483540534973], [-0.42806440591812134, 0.10874399542808533, -0.4567840099334717, -0.012656182050704956], [-0.2613893747329712, -0.07425457239151001, 0.02400289475917816, 0.299114465713501]], dtype='float32').reshape([7, 4]),
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


class TestPrimitiveOp_6b587d0122cdf4e52b38deacc5574df7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0744b88908a158c9736c500ad1d8b491
    def get_inputs(self):
        return [
            paddle.uniform([4618, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e9b7837f2d80554856bba3d24b2b94de(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0744b88908a158c9736c500ad1d8b491
    def get_inputs(self):
        return [
            paddle.uniform([1058, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e8af3af1622f8eb02aa57fd2d2bf74a3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_44ae888f422ed4f4229b2aa25a9ec133
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.13133330643177032, 0.08482658863067627, -0.2580347955226898, -0.09818729758262634], [0.10038453340530396, 0.046277258545160294, 0.16053301095962524, -0.016694992780685425], [0.10038453340530396, 0.046277258545160294, 0.16053301095962524, -0.016694992780685425], [0.0273551344871521, -0.002341151237487793, -0.11530663818120956, 0.42686018347740173], [-0.03745202720165253, -0.1409778594970703, 0.37401413917541504, 0.4199698865413666], [0.12621213495731354, -0.11907048523426056, -0.40092790126800537, 0.29637330770492554]], dtype='float32').reshape([6, 4]),
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


class TestPrimitiveOp_7a830c4d53fb2fd93128d3fdd66c44c7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0744b88908a158c9736c500ad1d8b491
    def get_inputs(self):
        return [
            paddle.uniform([2402, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e226d45e04e8c78d6425b7b86379839c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0744b88908a158c9736c500ad1d8b491
    def get_inputs(self):
        return [
            paddle.uniform([2993, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_0a81220d6591ae7e94f28f70266b096b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0744b88908a158c9736c500ad1d8b491
    def get_inputs(self):
        return [
            paddle.uniform([3787, 4], dtype='float32', min=0, max=0.5),
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


class TestPrimitiveOp_c7f3d0f96b1240e7bd2004a9a804fc79(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0744b88908a158c9736c500ad1d8b491
    def get_inputs(self):
        return [
            paddle.uniform([2114, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_5b554c694ee987feeaa3318114089415(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_935aa3778f50ab3eef1173d72ce6f082
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 32, 32], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a231dc9d868de5cdf7f1d17d0c4393a8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_44ae888f422ed4f4229b2aa25a9ec133
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.016613949090242386, 0.35582423210144043, -0.07691299915313721, 0.14768289029598236], [0.03833845257759094, 0.2901633381843567, 0.2641144394874573, -0.13769707083702087], [0.012129008769989014, -0.18745338916778564, -0.14885768294334412, 0.18658161163330078], [0.012129008769989014, -0.18745338916778564, -0.14885768294334412, 0.18658161163330078], [0.2790951132774353, -0.014318227767944336, 0.13249212503433228, -0.14775767922401428]], dtype='float32').reshape([5, 4]),
        ]


class TestPrimitiveOp_4eee3c2f51c339b160066afba57fa6db(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_44ae888f422ed4f4229b2aa25a9ec133
    def get_inputs(self):
        return [
            paddle.uniform([56, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3168888922b7c6c640b163a776e0cb55(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0744b88908a158c9736c500ad1d8b491
    def get_inputs(self):
        return [
            paddle.uniform([4156, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_2c6a7cbaa0f51f9724390ed3eb425718(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_44ae888f422ed4f4229b2aa25a9ec133
    def get_inputs(self):
        return [
            paddle.to_tensor([[-0.1323268711566925, -0.23802849650382996, -0.3332147002220154, 0.11582101136445999], [0.013273239135742188, 0.00046723615378141403, -0.01889728009700775, -0.39437851309776306], [-0.16595038771629333, -0.00343361496925354, 0.23501847684383392, -0.11366520822048187], [-0.1323268711566925, -0.23802849650382996, -0.3332147002220154, 0.11582101136445999], [-0.05888435244560242, 0.15700788795948029, -0.1641436368227005, -0.04239286482334137], [-0.1870676726102829, 0.07733932137489319, 0.09762038290500641, -0.06075166165828705], [-0.05888435244560242, 0.15700788795948029, -0.1641436368227005, -0.04239286482334137]], dtype='float32').reshape([7, 4]),
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