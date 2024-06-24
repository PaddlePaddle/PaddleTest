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


class TestPrimitiveOp_d873bfa3960495a627d6ceef37040347(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_44ae888f422ed4f4229b2aa25a9ec133
    def get_inputs(self):
        return [
            paddle.uniform([1745, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_23c8b25b7d0a9ef31f5688828cf21442(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_44ae888f422ed4f4229b2aa25a9ec133
    def get_inputs(self):
        return [
            paddle.to_tensor([[-0.23606227338314056, 0.1306420862674713, -0.056651972234249115, 0.15376406908035278], [0.0829421803355217, -0.21722771227359772, 0.06370577216148376, 0.11743849515914917], [0.32634949684143066, 0.11071017384529114, 0.04787001013755798, 0.39140892028808594], [0.34305647015571594, -0.3147853910923004, -0.21061715483665466, 0.21905085444450378], [0.09009411931037903, -0.12927646934986115, -0.13630236685276031, -0.38561052083969116]], dtype='float32').reshape([5, 4]),
        ]


class TestPrimitiveOp_228f6dcabfc3fd68067e718cb4198937(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_44ae888f422ed4f4229b2aa25a9ec133
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.10176767408847809, -0.027969181537628174, 0.13815650343894958, 0.07590332627296448], [-0.3440432548522949, -0.017140405252575874, -0.02755213901400566, -0.09486806392669678], [0.1388358175754547, -0.20331719517707825, 0.34952691197395325, -0.289692759513855], [-0.3440432548522949, -0.017140405252575874, -0.02755213901400566, -0.09486806392669678], [0.1388358175754547, -0.20331719517707825, 0.34952691197395325, -0.289692759513855]], dtype='float32').reshape([5, 4]),
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


class TestPrimitiveOp_6570992653dbfe842e161fdee85dc70f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_44ae888f422ed4f4229b2aa25a9ec133
    def get_inputs(self):
        return [
            paddle.uniform([5556, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_0e62c03ad9fa899fc23e3eb7faec43a5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_44ae888f422ed4f4229b2aa25a9ec133
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.1511559784412384, 0.16333091259002686, 0.000880807638168335, -0.04097789525985718], [0.06759227812290192, -0.11715888977050781, -0.005081087350845337, 0.07123449444770813], [-0.0758233442902565, -0.31037062406539917, 0.3314056992530823, 0.09279140830039978], [0.06759227812290192, -0.11715888977050781, -0.005081087350845337, 0.07123449444770813], [-0.0758233442902565, -0.31037062406539917, 0.3314056992530823, 0.09279140830039978], [-0.06419932842254639, -0.023158907890319824, -0.032714784145355225, -0.24787023663520813], [-0.06419932842254639, -0.023158907890319824, -0.032714784145355225, -0.24787023663520813]], dtype='float32').reshape([7, 4]),
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


class TestPrimitiveOp_11e7b3b44e1f5b49407d4fe2243005c6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_44ae888f422ed4f4229b2aa25a9ec133
    def get_inputs(self):
        return [
            paddle.uniform([1744, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_2e2a4348030b3672d84f71a6a01ad8cb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_44ae888f422ed4f4229b2aa25a9ec133
    def get_inputs(self):
        return [
            paddle.uniform([1547, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_924c769897ea08a49831d7eb452a5094(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_44ae888f422ed4f4229b2aa25a9ec133
    def get_inputs(self):
        return [
            paddle.to_tensor([[-0.03431218862533569, -0.026114583015441895, -0.3125544786453247, 0.05361819267272949], [0.2707841992378235, 0.19268369674682617, -0.006156504154205322, 0.09308546781539917], [0.10481724143028259, 0.22131045162677765, 0.2644655704498291, 0.28144195675849915], [0.18018700182437897, -0.11127512902021408, -0.4410821199417114, 0.0970020592212677], [0.18018700182437897, -0.11127512902021408, -0.4410821199417114, 0.0970020592212677], [0.10481724143028259, 0.22131045162677765, 0.2644655704498291, 0.28144195675849915]], dtype='float32').reshape([6, 4]),
        ]


class TestPrimitiveOp_7c8091579df4dddb3f6a0aadc923c147(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_44ae888f422ed4f4229b2aa25a9ec133
    def get_inputs(self):
        return [
            paddle.to_tensor([[-0.202517569065094, 0.2943516671657562, 0.30916333198547363, -0.01913774013519287], [-0.2773173451423645, 0.03562504053115845, 0.11010816693305969, -0.3304157257080078], [0.3562518060207367, -0.21646155416965485, -0.03921368718147278, 0.10079234093427658], [0.05663609504699707, -0.09692704677581787, -0.04421214759349823, 0.12598776817321777], [-0.202517569065094, 0.2943516671657562, 0.30916333198547363, -0.01913774013519287]], dtype='float32').reshape([5, 4]),
        ]


class TestPrimitiveOp_f7ee127404489ac713dd64b92089fd22(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_44ae888f422ed4f4229b2aa25a9ec133
    def get_inputs(self):
        return [
            paddle.uniform([10, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9123940c1bbd0f4212d2bccc52976ff6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_44ae888f422ed4f4229b2aa25a9ec133
    def get_inputs(self):
        return [
            paddle.to_tensor([[-0.36131948232650757, 0.3353999853134155, -0.02784162014722824, 0.07963600009679794], [0.41048651933670044, -0.3893478214740753, -0.2992238700389862, 0.15474331378936768], [-0.08667996525764465, 0.19611337780952454, 0.06163033843040466, 0.023111797869205475], [-0.1219969391822815, 0.08688667416572571, -0.3911390006542206, 0.2473030537366867]], dtype='float32').reshape([4, 4]),
        ]


class TestPrimitiveOp_748a0fcd9c8c1b0162066c1dc314b930(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_44ae888f422ed4f4229b2aa25a9ec133
    def get_inputs(self):
        return [
            paddle.uniform([84, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_28055a9161ccd42d6ae82ea8cfd6cd9b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_44ae888f422ed4f4229b2aa25a9ec133
    def get_inputs(self):
        return [
            paddle.uniform([2056, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_08a5e8ed06aa7df6d37021e8a4020587(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_44ae888f422ed4f4229b2aa25a9ec133
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.3118785619735718, -0.06509244441986084, -0.29392749071121216, -0.3088516592979431], [0.3118785619735718, -0.06509244441986084, -0.29392749071121216, -0.3088516592979431], [-0.16390565037727356, -0.3733579218387604, 0.2686590552330017, 0.21544930338859558], [0.2529969811439514, 0.006404057145118713, -0.19852834939956665, 0.1327028125524521], [-0.0964447557926178, -0.10411974787712097, -0.06578028202056885, 0.07751882076263428], [-0.4481545686721802, -0.07796558737754822, 0.0771525502204895, 0.06475737690925598], [-0.20786544680595398, -0.2283610701560974, 0.08375959098339081, 0.08007444441318512]], dtype='float32').reshape([7, 4]),
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


class TestPrimitiveOp_372d35c9c8d2b45d949397605a81cb8c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_44ae888f422ed4f4229b2aa25a9ec133
    def get_inputs(self):
        return [
            paddle.uniform([4650, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_7efe382f3f23e477044c4835797892b9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_44ae888f422ed4f4229b2aa25a9ec133
    def get_inputs(self):
        return [
            paddle.uniform([1059, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b315371f35b6fda69d8e197c57209296(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_44ae888f422ed4f4229b2aa25a9ec133
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.07373276352882385, 0.1829991638660431, -0.21339315176010132, -0.003011345863342285], [-0.028881244361400604, -0.03423634171485901, -0.43788984417915344, 0.17044034600257874], [-0.028881244361400604, -0.03423634171485901, -0.43788984417915344, 0.17044034600257874], [0.13281306624412537, 0.10249564051628113, 0.12424007803201675, -0.47744619846343994], [0.18827302753925323, 0.06766808032989502, 0.022206313908100128, 0.06729784607887268], [0.1352621465921402, 0.3263186514377594, 0.1006387323141098, -0.21034857630729675]], dtype='float32').reshape([6, 4]),
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


class TestPrimitiveOp_56a059527bcfb0f76e69bef67fa5bb1d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_44ae888f422ed4f4229b2aa25a9ec133
    def get_inputs(self):
        return [
            paddle.uniform([2347, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_21c54b3a92ad81fd7d77b70003ffb88a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_44ae888f422ed4f4229b2aa25a9ec133
    def get_inputs(self):
        return [
            paddle.uniform([3109, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_84398d30a74964fd988f1a77b7fd07a9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_44ae888f422ed4f4229b2aa25a9ec133
    def get_inputs(self):
        return [
            paddle.uniform([3813, 4], dtype='float32', min=0, max=0.5),
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


class TestPrimitiveOp_5025bac6ce589dafc61d4bfe21656add(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_44ae888f422ed4f4229b2aa25a9ec133
    def get_inputs(self):
        return [
            paddle.uniform([2100, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b6a1f358aa54b725cdd34967e0e3bb6a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_135494904ccd14f6906343640e2dc1f3
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 32, 32], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ab440e2d93f7e3defbe84f074d57df05(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_44ae888f422ed4f4229b2aa25a9ec133
    def get_inputs(self):
        return [
            paddle.to_tensor([[-0.3232983648777008, -0.26429325342178345, 0.032824814319610596, -0.06961558759212494], [0.10855147242546082, 0.23897580802440643, 0.12071990966796875, -0.02904871106147766], [0.039555326104164124, 0.10271425545215607, 0.11992594599723816, 0.08371031284332275], [0.039555326104164124, 0.10271425545215607, 0.11992594599723816, 0.08371031284332275], [0.25436314940452576, 0.24295449256896973, 0.14639168977737427, -0.03259342908859253]], dtype='float32').reshape([5, 4]),
        ]


class TestPrimitiveOp_4eee3c2f51c339b160066afba57fa6db(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_44ae888f422ed4f4229b2aa25a9ec133
    def get_inputs(self):
        return [
            paddle.uniform([56, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f8c3de42e9b7fadd3cb1c2571cdf9b68(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_44ae888f422ed4f4229b2aa25a9ec133
    def get_inputs(self):
        return [
            paddle.uniform([4231, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_5027ea50e10767c5a283947ee3604eea(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_44ae888f422ed4f4229b2aa25a9ec133
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.19802163541316986, -0.16961577534675598, 0.15227729082107544, 0.025044582784175873], [0.2353224754333496, 0.05387616157531738, -0.14789704978466034, 0.0630868673324585], [-0.001409083604812622, -0.05223914980888367, 0.08461788296699524, -0.06864841282367706], [0.19802163541316986, -0.16961577534675598, 0.15227729082107544, 0.025044582784175873], [0.03428974747657776, -0.11960519850254059, -0.20580589771270752, -0.07139194011688232], [-0.0854860246181488, 0.07616549730300903, 0.29124677181243896, -0.13023613393306732], [0.03428974747657776, -0.11960519850254059, -0.20580589771270752, -0.07139194011688232]], dtype='float32').reshape([7, 4]),
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