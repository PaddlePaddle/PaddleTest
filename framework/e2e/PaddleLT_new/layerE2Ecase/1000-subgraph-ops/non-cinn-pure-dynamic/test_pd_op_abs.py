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


class TestPrimitiveOp_612fcbd29fde44888fb4d4fcc8ffd169(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_44ae888f422ed4f4229b2aa25a9ec133
    def get_inputs(self):
        return [
            paddle.uniform([1762, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d59f4c59f96853f2e2ecb27187a996a7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_44ae888f422ed4f4229b2aa25a9ec133
    def get_inputs(self):
        return [
            paddle.to_tensor([[-0.06151318550109863, -0.11936834454536438, 0.07800794392824173, -0.267497181892395], [0.11394819617271423, 0.005255617201328278, -0.09603127837181091, 0.057251498103141785], [0.2812342643737793, 0.2772517204284668, 0.2135857790708542, -0.19352394342422485], [-0.05876466631889343, -0.2864294648170471, -0.11026737093925476, -0.3292335867881775], [0.22799256443977356, -0.13594946265220642, -0.23514008522033691, 0.044174402952194214]], dtype='float32').reshape([5, 4]),
        ]


class TestPrimitiveOp_0bf8a366aa58aaacc516d01dbc09a6f2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_44ae888f422ed4f4229b2aa25a9ec133
    def get_inputs(self):
        return [
            paddle.to_tensor([[-0.26748061180114746, -0.13443715870380402, 0.014436036348342896, -0.42332661151885986], [-0.11801926791667938, 0.1759794056415558, -0.27236711978912354, -0.30360615253448486], [0.06403589248657227, -0.21310356259346008, -0.02465561032295227, 0.3234819173812866], [-0.11801926791667938, 0.1759794056415558, -0.27236711978912354, -0.30360615253448486], [0.06403589248657227, -0.21310356259346008, -0.02465561032295227, 0.3234819173812866]], dtype='float32').reshape([5, 4]),
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


class TestPrimitiveOp_f0d372613ceede39e00e8b7f2c24f6b3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_44ae888f422ed4f4229b2aa25a9ec133
    def get_inputs(self):
        return [
            paddle.uniform([5522, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_1b7b45b585e3f3e1e1494c6196097ea6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_44ae888f422ed4f4229b2aa25a9ec133
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.06942878663539886, -0.39978697896003723, -0.04090121388435364, 0.11602064967155457], [0.0561792254447937, 0.21276485919952393, -0.11734984070062637, 0.08785513043403625], [0.1648823320865631, -0.051890671253204346, 0.014645934104919434, 0.06937026977539062], [0.0561792254447937, 0.21276485919952393, -0.11734984070062637, 0.08785513043403625], [0.1648823320865631, -0.051890671253204346, 0.014645934104919434, 0.06937026977539062], [-0.20225289463996887, 0.25074583292007446, 0.2275557518005371, -0.05374142527580261], [-0.20225289463996887, 0.25074583292007446, 0.2275557518005371, -0.05374142527580261]], dtype='float32').reshape([7, 4]),
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


class TestPrimitiveOp_e65fdd17f4372850a6f53e0fefb48f9f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_44ae888f422ed4f4229b2aa25a9ec133
    def get_inputs(self):
        return [
            paddle.uniform([1760, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_1ce8d00aecc14f539d9ebbd82f6c602f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_44ae888f422ed4f4229b2aa25a9ec133
    def get_inputs(self):
        return [
            paddle.uniform([1522, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_2ba3f2a34166c8693f48d341c7f1f7a5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_44ae888f422ed4f4229b2aa25a9ec133
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.10981824994087219, 0.1818036437034607, 0.005384296178817749, -0.17434953153133392], [0.12394212931394577, -0.34773626923561096, 0.10430505871772766, 0.23078596591949463], [0.2615264356136322, 0.13417835533618927, 0.25303182005882263, 0.0923982560634613], [-0.4147195816040039, 0.29263797402381897, -0.10034643113613129, -0.3984808325767517], [-0.4147195816040039, 0.29263797402381897, -0.10034643113613129, -0.3984808325767517], [0.2615264356136322, 0.13417835533618927, 0.25303182005882263, 0.0923982560634613]], dtype='float32').reshape([6, 4]),
        ]


class TestPrimitiveOp_29654151660b190487cc212ca6ba47e5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_44ae888f422ed4f4229b2aa25a9ec133
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.1289084106683731, 0.24571344256401062, 0.264828622341156, 0.1877102255821228], [0.04371963441371918, -0.03634536266326904, 0.05285709723830223, 0.056930214166641235], [-0.34693601727485657, -0.18697671592235565, -0.08248449862003326, -0.23773470520973206], [0.18576744198799133, 0.06885099411010742, -0.14518606662750244, 0.03412967920303345], [0.1289084106683731, 0.24571344256401062, 0.264828622341156, 0.1877102255821228]], dtype='float32').reshape([5, 4]),
        ]


class TestPrimitiveOp_f7ee127404489ac713dd64b92089fd22(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_44ae888f422ed4f4229b2aa25a9ec133
    def get_inputs(self):
        return [
            paddle.uniform([10, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_281831292d099b17225df7eac3f77ca9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_44ae888f422ed4f4229b2aa25a9ec133
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.030229568481445312, 0.15371039509773254, 0.21909597516059875, -0.0034876130521297455], [0.09754230082035065, -0.1374218612909317, 0.24284592270851135, 0.06160402297973633], [-0.06366714835166931, 0.1073785126209259, 0.3071011006832123, 0.03405541181564331], [-0.09629049897193909, -0.00018036365509033203, -0.1322488784790039, 0.40136411786079407]], dtype='float32').reshape([4, 4]),
        ]


class TestPrimitiveOp_748a0fcd9c8c1b0162066c1dc314b930(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_44ae888f422ed4f4229b2aa25a9ec133
    def get_inputs(self):
        return [
            paddle.uniform([84, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f04c96f80716dc5f71fa03d89f63b93e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_44ae888f422ed4f4229b2aa25a9ec133
    def get_inputs(self):
        return [
            paddle.uniform([2074, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_027a7dfe0e192ee74403f26cc7ccf890(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_44ae888f422ed4f4229b2aa25a9ec133
    def get_inputs(self):
        return [
            paddle.to_tensor([[-0.17910704016685486, 0.3516285717487335, -0.11908003687858582, -0.24569259583950043], [-0.17910704016685486, 0.3516285717487335, -0.11908003687858582, -0.24569259583950043], [0.02250000834465027, 0.35152024030685425, -0.15124770998954773, 0.09407028555870056], [0.16901521384716034, 0.23902814090251923, 0.1676264852285385, -0.28854885697364807], [-0.1521880328655243, 0.27202269434928894, -0.07244840264320374, 0.10833775997161865], [0.13918226957321167, -0.06347241997718811, 0.27488210797309875, 0.3045266270637512], [-0.02585412561893463, -0.12091132253408432, -0.24972589313983917, -0.03646087646484375]], dtype='float32').reshape([7, 4]),
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


class TestPrimitiveOp_0141b6e29dc9ebe7ea02ef2424b6ad4b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_44ae888f422ed4f4229b2aa25a9ec133
    def get_inputs(self):
        return [
            paddle.uniform([4734, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f0e2e473565d21981b61724951cab76e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_44ae888f422ed4f4229b2aa25a9ec133
    def get_inputs(self):
        return [
            paddle.uniform([1074, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_84c5789cf454d1b0d99f467c076032c4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_44ae888f422ed4f4229b2aa25a9ec133
    def get_inputs(self):
        return [
            paddle.to_tensor([[-0.17867141962051392, -0.038635507225990295, -0.10939210653305054, 0.22395551204681396], [0.01621919870376587, 0.3006782531738281, -0.033404335379600525, -0.06576906144618988], [0.01621919870376587, 0.3006782531738281, -0.033404335379600525, -0.06576906144618988], [-0.2828966975212097, -0.11116990447044373, 0.3995091915130615, 0.10126431286334991], [-0.2237587720155716, -0.027776330709457397, -0.017031311988830566, 0.014228790998458862], [0.1819733828306198, 0.3137103319168091, 0.27327537536621094, -0.2752225995063782]], dtype='float32').reshape([6, 4]),
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


class TestPrimitiveOp_97d4c8be6c2dd096e7ea4cfaf6538a05(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_44ae888f422ed4f4229b2aa25a9ec133
    def get_inputs(self):
        return [
            paddle.uniform([2332, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_1701f426642a35865e3070b63cb5dd79(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_44ae888f422ed4f4229b2aa25a9ec133
    def get_inputs(self):
        return [
            paddle.uniform([3051, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3e9e91505f1d0c5a6ac73956e6b6c60d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_44ae888f422ed4f4229b2aa25a9ec133
    def get_inputs(self):
        return [
            paddle.uniform([3870, 4], dtype='float32', min=0, max=0.5),
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


class TestPrimitiveOp_b26f1f292f294c029a2f52465f419fc7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_44ae888f422ed4f4229b2aa25a9ec133
    def get_inputs(self):
        return [
            paddle.uniform([2111, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b6a1f358aa54b725cdd34967e0e3bb6a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_135494904ccd14f6906343640e2dc1f3
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 32, 32], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6a2402b1497555822453918c27e31f48(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_44ae888f422ed4f4229b2aa25a9ec133
    def get_inputs(self):
        return [
            paddle.to_tensor([[-0.13473466038703918, -0.12285895645618439, 0.10564139485359192, 0.24786412715911865], [-0.34976115822792053, -0.31551921367645264, -0.26948869228363037, 0.12118439376354218], [0.2146769016981125, -0.40058374404907227, -0.1350015103816986, -0.13491290807724], [0.2146769016981125, -0.40058374404907227, -0.1350015103816986, -0.13491290807724], [-0.0060720741748809814, -0.19148385524749756, -0.026892781257629395, 0.30257436633110046]], dtype='float32').reshape([5, 4]),
        ]


class TestPrimitiveOp_4eee3c2f51c339b160066afba57fa6db(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_44ae888f422ed4f4229b2aa25a9ec133
    def get_inputs(self):
        return [
            paddle.uniform([56, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_839dc9078c658c34119cf89c0d01e498(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_44ae888f422ed4f4229b2aa25a9ec133
    def get_inputs(self):
        return [
            paddle.uniform([4141, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_0b20d12f786dbc927f4147247e1f87ec(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_44ae888f422ed4f4229b2aa25a9ec133
    def get_inputs(self):
        return [
            paddle.to_tensor([[-0.14209389686584473, -0.06900376081466675, 0.2527446746826172, -0.19547338783740997], [0.03495118021965027, 0.08211711049079895, 0.1606784164905548, 0.3424886167049408], [0.0032216906547546387, 0.25409072637557983, -0.05634039640426636, -0.21551376581192017], [-0.14209389686584473, -0.06900376081466675, 0.2527446746826172, -0.19547338783740997], [0.48535096645355225, 0.050888895988464355, 0.054100051522254944, 0.11393816769123077], [-0.10485643148422241, 0.06027994304895401, 0.3643565773963928, -0.058671534061431885], [0.48535096645355225, 0.050888895988464355, 0.054100051522254944, 0.11393816769123077]], dtype='float32').reshape([7, 4]),
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