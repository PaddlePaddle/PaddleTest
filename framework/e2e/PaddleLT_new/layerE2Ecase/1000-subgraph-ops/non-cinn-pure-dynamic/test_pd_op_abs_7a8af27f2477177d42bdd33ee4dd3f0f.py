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


class TestPrimitiveOp_87c76be8bef79ed3d70601f156c386f3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_44ae888f422ed4f4229b2aa25a9ec133
    def get_inputs(self):
        return [
            paddle.uniform([1787, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_649ff4b46418f3ba9c94a58858bf1a49(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_44ae888f422ed4f4229b2aa25a9ec133
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.08987224102020264, 0.09446462988853455, -0.0645132064819336, -0.09983265399932861], [-0.09016704559326172, -0.1352386474609375, -0.2301655262708664, -0.1461450159549713], [0.21332000195980072, 0.036309823393821716, -0.010198406875133514, 0.40485113859176636], [-0.1312466859817505, 0.13634832203388214, -0.39518219232559204, 0.09201464056968689], [-0.06631225347518921, 0.054139167070388794, -0.10006687045097351, 0.3483027219772339]], dtype='float32').reshape([5, 4]),
        ]


class TestPrimitiveOp_23e3203358bf27b16e9818c472a07ce7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_44ae888f422ed4f4229b2aa25a9ec133
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.15815213322639465, 0.08740302920341492, -0.0651894211769104, -0.28641700744628906], [0.016894638538360596, 0.2502749264240265, -0.12896551191806793, 0.2630677819252014], [0.23832650482654572, -0.2584977149963379, 0.0639914721250534, 0.21149897575378418], [0.016894638538360596, 0.2502749264240265, -0.12896551191806793, 0.2630677819252014], [0.23832650482654572, -0.2584977149963379, 0.0639914721250534, 0.21149897575378418]], dtype='float32').reshape([5, 4]),
        ]


class PrimitiveOp_135494904ccd14f6906343640e2dc1f3(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.abs(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_cf17457ebab6321b9f9d58ad2e337de2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_135494904ccd14f6906343640e2dc1f3
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 8, 8], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_0510014e2fc94748134583192f6fe5f3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_44ae888f422ed4f4229b2aa25a9ec133
    def get_inputs(self):
        return [
            paddle.uniform([5524, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_241bd4ee8b8c8ee40d84afb6cd82476f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_44ae888f422ed4f4229b2aa25a9ec133
    def get_inputs(self):
        return [
            paddle.to_tensor([[-0.08980125188827515, 0.10924512147903442, 0.013537660241127014, 0.30076491832733154], [0.14479558169841766, -0.2556632459163666, 0.11069628596305847, -0.25762078166007996], [0.18983790278434753, 0.15936079621315002, 0.4752041697502136, -0.008670181035995483], [0.14479558169841766, -0.2556632459163666, 0.11069628596305847, -0.25762078166007996], [0.18983790278434753, 0.15936079621315002, 0.4752041697502136, -0.008670181035995483], [0.07695235311985016, -0.12814626097679138, 0.24936996400356293, 0.28386378288269043], [0.07695235311985016, -0.12814626097679138, 0.24936996400356293, 0.28386378288269043]], dtype='float32').reshape([7, 4]),
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


class TestPrimitiveOp_10f120163ed2d1a7af0e77320b7d6a20(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_44ae888f422ed4f4229b2aa25a9ec133
    def get_inputs(self):
        return [
            paddle.uniform([1722, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b224fba25af13ef56d83644a25530b11(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_44ae888f422ed4f4229b2aa25a9ec133
    def get_inputs(self):
        return [
            paddle.uniform([1565, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_988685cbb17dc7a3dd0e88effef7f4fc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_44ae888f422ed4f4229b2aa25a9ec133
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.07388460636138916, 0.34592828154563904, 0.13610978424549103, 0.030362337827682495], [-0.14449560642242432, 0.15574820339679718, 0.3628971576690674, -0.059735897928476334], [0.22425659000873566, 0.20642846822738647, 0.19675441086292267, 0.21475961804389954], [-0.04178471863269806, -0.07089582085609436, 0.4266142249107361, 0.2633800506591797], [-0.04178471863269806, -0.07089582085609436, 0.4266142249107361, 0.2633800506591797], [0.22425659000873566, 0.20642846822738647, 0.19675441086292267, 0.21475961804389954]], dtype='float32').reshape([6, 4]),
        ]


class TestPrimitiveOp_484176acba046269ff716a0c856dec14(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_44ae888f422ed4f4229b2aa25a9ec133
    def get_inputs(self):
        return [
            paddle.to_tensor([[-0.2612953782081604, -0.28655803203582764, -0.29477638006210327, 0.2740551829338074], [-0.07684624195098877, 0.028802603483200073, -0.05022908002138138, 0.10331213474273682], [-0.09165571630001068, -0.16482195258140564, 0.15293759107589722, 0.1458241045475006], [-0.23540610074996948, 0.005343496799468994, -0.14588503539562225, 0.05192685127258301], [-0.2612953782081604, -0.28655803203582764, -0.29477638006210327, 0.2740551829338074]], dtype='float32').reshape([5, 4]),
        ]


class TestPrimitiveOp_f7ee127404489ac713dd64b92089fd22(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_44ae888f422ed4f4229b2aa25a9ec133
    def get_inputs(self):
        return [
            paddle.uniform([10, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_927ddf5d3db4e51d6ee8b0d74f3ab425(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_44ae888f422ed4f4229b2aa25a9ec133
    def get_inputs(self):
        return [
            paddle.to_tensor([[-0.38793617486953735, -0.0617041140794754, 0.16063280403614044, 0.06832501292228699], [-0.14784809947013855, -0.2535051703453064, -0.0055626630783081055, -0.2679281234741211], [-0.2185918092727661, 0.24922549724578857, 0.26665961742401123, -0.18001899123191833], [-0.19315892457962036, -0.292204886674881, -0.23457635939121246, 0.3767058849334717]], dtype='float32').reshape([4, 4]),
        ]


class TestPrimitiveOp_748a0fcd9c8c1b0162066c1dc314b930(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_44ae888f422ed4f4229b2aa25a9ec133
    def get_inputs(self):
        return [
            paddle.uniform([84, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_cbc12472d54050a974f3459290691904(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_44ae888f422ed4f4229b2aa25a9ec133
    def get_inputs(self):
        return [
            paddle.uniform([2034, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c2664fd5f83012673fe078f0ecb6a344(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_44ae888f422ed4f4229b2aa25a9ec133
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.17936022579669952, -0.029575109481811523, -0.3104240298271179, -0.019696682691574097], [0.17936022579669952, -0.029575109481811523, -0.3104240298271179, -0.019696682691574097], [-0.2813943028450012, 0.1955566108226776, 0.018820174038410187, -0.11219587922096252], [0.0066500455141067505, -0.046088606119155884, -0.012532830238342285, -0.1825047880411148], [0.35801857709884644, 0.08647437393665314, 0.11524057388305664, -0.3932095468044281], [0.08881855010986328, 0.07655680179595947, 0.21666808426380157, -0.44117167592048645], [0.3748916983604431, -0.12003204226493835, -0.022046446800231934, 0.014015290886163712]], dtype='float32').reshape([7, 4]),
        ]


class TestPrimitiveOp_b1528af25a99d7463a71cc62d868ad58(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_44ae888f422ed4f4229b2aa25a9ec133
    def get_inputs(self):
        return [
            paddle.uniform([16384, 5], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f2984390a495576bc365e7a80af30ecf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_135494904ccd14f6906343640e2dc1f3
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 64, 64], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_2305a50e4fe3e8a8b8992372c2a5e1e2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_44ae888f422ed4f4229b2aa25a9ec133
    def get_inputs(self):
        return [
            paddle.uniform([4667, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4c8b62275dc3ecfe6e5faa0b1ab96d89(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_44ae888f422ed4f4229b2aa25a9ec133
    def get_inputs(self):
        return [
            paddle.uniform([1052, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ddc0413d25c50e1bcf68bcc2ea1a1752(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_44ae888f422ed4f4229b2aa25a9ec133
    def get_inputs(self):
        return [
            paddle.to_tensor([[-0.06581726670265198, -0.012930139899253845, -0.31881576776504517, 0.04646962881088257], [0.40379393100738525, 0.012253254652023315, -0.35942530632019043, 0.08588998019695282], [0.40379393100738525, 0.012253254652023315, -0.35942530632019043, 0.08588998019695282], [0.08568122982978821, 0.09530001878738403, -0.22064247727394104, -0.42547687888145447], [0.3612695634365082, -0.2549719512462616, -0.25649890303611755, 0.2786141335964203], [0.14194762706756592, 0.3295552432537079, -0.39786407351493835, -0.16491538286209106]], dtype='float32').reshape([6, 4]),
        ]


class PrimitiveOp_e1b9d219a1e72b6bd4b731f1be18a8ae(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.abs(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_95676990489c26fcc42f9414185f4ac0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e1b9d219a1e72b6bd4b731f1be18a8ae
    def get_inputs(self):
        return [
            paddle.uniform([100, 2, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e10327b96d83b1b8fe320e066502da8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e1b9d219a1e72b6bd4b731f1be18a8ae
    def get_inputs(self):
        return [
            paddle.uniform([300, 2, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6c3b57b320b12d6c04750cea87315b95(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_135494904ccd14f6906343640e2dc1f3
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 128, 128], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_171ed77778faebc5c01169ef0ddf13ff(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_44ae888f422ed4f4229b2aa25a9ec133
    def get_inputs(self):
        return [
            paddle.uniform([2378, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_81b5d9d62ac84292fc7343b26588e579(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_44ae888f422ed4f4229b2aa25a9ec133
    def get_inputs(self):
        return [
            paddle.uniform([3105, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b93365f7f662e307dfcbc3f25b68f001(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_44ae888f422ed4f4229b2aa25a9ec133
    def get_inputs(self):
        return [
            paddle.uniform([3832, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a82622db00f01ccee5196c226de42d97(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_135494904ccd14f6906343640e2dc1f3
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


class TestPrimitiveOp_506303fdcda504e0251e92cc4b1d04c7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_44ae888f422ed4f4229b2aa25a9ec133
    def get_inputs(self):
        return [
            paddle.uniform([2087, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b6a1f358aa54b725cdd34967e0e3bb6a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_135494904ccd14f6906343640e2dc1f3
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 32, 32], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9160142f407b6a060e79b9368cd484e9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_44ae888f422ed4f4229b2aa25a9ec133
    def get_inputs(self):
        return [
            paddle.to_tensor([[-0.08812366425991058, -0.028453081846237183, -0.10375607013702393, -0.09819439053535461], [0.15779852867126465, -0.016703754663467407, 0.062226682901382446, 0.12866711616516113], [0.14701034128665924, 0.0776543989777565, -0.2557470202445984, -0.01866358518600464], [0.14701034128665924, 0.0776543989777565, -0.2557470202445984, -0.01866358518600464], [0.2722809612751007, -0.014804542064666748, -0.3420112133026123, 0.3253910541534424]], dtype='float32').reshape([5, 4]),
        ]


class TestPrimitiveOp_4eee3c2f51c339b160066afba57fa6db(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_44ae888f422ed4f4229b2aa25a9ec133
    def get_inputs(self):
        return [
            paddle.uniform([56, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a83220b772573bcd9e6345b3bd8abbb2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_44ae888f422ed4f4229b2aa25a9ec133
    def get_inputs(self):
        return [
            paddle.uniform([4271, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_dc7f1007f402ba9fde6e22430c181443(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_44ae888f422ed4f4229b2aa25a9ec133
    def get_inputs(self):
        return [
            paddle.to_tensor([[-0.17719396948814392, 0.0036325454711914062, 0.1302030086517334, -0.29399293661117554], [0.25933337211608887, -0.26464760303497314, 0.024477168917655945, 0.1406756043434143], [0.08580110222101212, 0.012520700693130493, -0.13238537311553955, -0.24548061192035675], [-0.17719396948814392, 0.0036325454711914062, 0.1302030086517334, -0.29399293661117554], [0.18552877008914948, -0.008648484945297241, 0.4290626347064972, -0.21437333524227142], [-0.026892393827438354, 0.03062039241194725, 0.22371087968349457, 0.15546290576457977], [0.18552877008914948, -0.008648484945297241, 0.4290626347064972, -0.21437333524227142]], dtype='float32').reshape([7, 4]),
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