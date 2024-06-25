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



class PrimitiveOp_f00b14e252dc77869bbcd461a640cec3(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.maximum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None, None, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_e964172c46ffdc33f808146cc0311e88(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f00b14e252dc77869bbcd461a640cec3
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e964172c46ffdc33f808146cc0311e88(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f00b14e252dc77869bbcd461a640cec3
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_bf2929b72dc2c817af734c7811bb283d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f00b14e252dc77869bbcd461a640cec3
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_bf2929b72dc2c817af734c7811bb283d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f00b14e252dc77869bbcd461a640cec3
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_47251a23ac117016f7e358b0d1d45e19(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f00b14e252dc77869bbcd461a640cec3
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_47251a23ac117016f7e358b0d1d45e19(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f00b14e252dc77869bbcd461a640cec3
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_2dc88c738ba8f4028e5c341aa7cf47f9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f00b14e252dc77869bbcd461a640cec3
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_2dc88c738ba8f4028e5c341aa7cf47f9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f00b14e252dc77869bbcd461a640cec3
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c92cbf8524e253b9523a281d87c603a6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f00b14e252dc77869bbcd461a640cec3
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c92cbf8524e253b9523a281d87c603a6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f00b14e252dc77869bbcd461a640cec3
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_1eb7a23dd374710aeca7d92950a807b0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f00b14e252dc77869bbcd461a640cec3
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_1eb7a23dd374710aeca7d92950a807b0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f00b14e252dc77869bbcd461a640cec3
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e964172c46ffdc33f808146cc0311e88(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f00b14e252dc77869bbcd461a640cec3
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e964172c46ffdc33f808146cc0311e88(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f00b14e252dc77869bbcd461a640cec3
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3f3dc7b350f9ad1862de696709cb6b25(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f00b14e252dc77869bbcd461a640cec3
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3f3dc7b350f9ad1862de696709cb6b25(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f00b14e252dc77869bbcd461a640cec3
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_8dd6488a8eb6a91ac7fd0f52405ef2df(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f00b14e252dc77869bbcd461a640cec3
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_8dd6488a8eb6a91ac7fd0f52405ef2df(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f00b14e252dc77869bbcd461a640cec3
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_791985e0863360b3b07d19d88c268b58(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.maximum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_085f2d2a3ef4b26b0220531d42d9b68d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_791985e0863360b3b07d19d88c268b58
    def get_inputs(self):
        return [
            paddle.uniform([1762, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1762, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_085f2d2a3ef4b26b0220531d42d9b68d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_791985e0863360b3b07d19d88c268b58
    def get_inputs(self):
        return [
            paddle.uniform([1762, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1762, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_085f2d2a3ef4b26b0220531d42d9b68d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_791985e0863360b3b07d19d88c268b58
    def get_inputs(self):
        return [
            paddle.uniform([1762, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1762, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_085f2d2a3ef4b26b0220531d42d9b68d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_791985e0863360b3b07d19d88c268b58
    def get_inputs(self):
        return [
            paddle.uniform([1762, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1762, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_5fab66a1508f07ff510e57a7abf0e931(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f00b14e252dc77869bbcd461a640cec3
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_5fab66a1508f07ff510e57a7abf0e931(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f00b14e252dc77869bbcd461a640cec3
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_32a2824b5753466216caf7b3a1b1fd89(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f00b14e252dc77869bbcd461a640cec3
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_32a2824b5753466216caf7b3a1b1fd89(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f00b14e252dc77869bbcd461a640cec3
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e0aa7dbeb1efb983f626189bbc76d586(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f00b14e252dc77869bbcd461a640cec3
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e0aa7dbeb1efb983f626189bbc76d586(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f00b14e252dc77869bbcd461a640cec3
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a20e0871324731e58726ec0dcc7e339d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f00b14e252dc77869bbcd461a640cec3
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a20e0871324731e58726ec0dcc7e339d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f00b14e252dc77869bbcd461a640cec3
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_914608f8c40ab55d7ef4f9686a878cc8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_791985e0863360b3b07d19d88c268b58
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.23177070915699005], [0.3827413022518158], [0.12799298763275146], [0.1969173699617386], [0.011041968129575253], [0.3928758203983307], [0.4579012989997864], [0.31710684299468994], [0.2874194085597992]], dtype='float32').reshape([9, 1]),
            paddle.to_tensor([[0.3858015835285187], [0.2665897011756897], [0.40329664945602417], [0.3532903492450714], [0.46032440662384033], [0.0526868961751461], [0.3947453498840332], [0.13757629692554474], [0.4311077892780304]], dtype='float32').reshape([9, 1]),
        ]


class TestPrimitiveOp_4ac25e70a1b5ef796122cf3727b40f6e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_791985e0863360b3b07d19d88c268b58
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.3669567406177521], [0.44098344445228577], [0.14103317260742188], [0.07160888612270355], [0.12475907802581787], [0.3803057074546814], [0.29980069398880005], [0.4812083840370178], [0.2672547399997711]], dtype='float32').reshape([9, 1]),
            paddle.to_tensor([[0.06200192868709564], [0.2388659417629242], [0.1773584485054016], [0.24322769045829773], [0.46047866344451904], [0.18422792851924896], [0.10528553277254105], [0.09368486702442169], [0.062429703772068024]], dtype='float32').reshape([9, 1]),
        ]


class TestPrimitiveOp_2098c21e60426295c4fc64c0b4b7e6b7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_791985e0863360b3b07d19d88c268b58
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.437514990568161], [0.08309690654277802], [0.46463990211486816], [0.015629051253199577], [0.41757383942604065], [0.012152628973126411], [0.48972272872924805], [0.30988749861717224], [0.16249637305736542]], dtype='float32').reshape([9, 1]),
            paddle.to_tensor([[0.1447327435016632], [0.1211131140589714], [0.038118865340948105], [0.29171764850616455], [0.4933195412158966], [0.26447802782058716], [0.4100273549556732], [0.4631510376930237], [0.04747486487030983]], dtype='float32').reshape([9, 1]),
        ]


class TestPrimitiveOp_feeea2992137b1fe457e3c70c1610a41(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_791985e0863360b3b07d19d88c268b58
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.03195398673415184], [0.4533325731754303], [0.38529640436172485], [0.4782434105873108], [0.19919998943805695], [0.4240538477897644], [0.22890320420265198], [0.41177159547805786], [0.28056761622428894]], dtype='float32').reshape([9, 1]),
            paddle.to_tensor([[0.0798618495464325], [0.20543742179870605], [0.4339882731437683], [0.4867236614227295], [0.4583660364151001], [0.29198157787323], [0.0029337117448449135], [0.12914401292800903], [0.013538859784603119]], dtype='float32').reshape([9, 1]),
        ]


class TestPrimitiveOp_8553dcbbbaea0506096effe6a6dbd925(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_791985e0863360b3b07d19d88c268b58
    def get_inputs(self):
        return [
            paddle.uniform([5522, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([5522, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_8553dcbbbaea0506096effe6a6dbd925(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_791985e0863360b3b07d19d88c268b58
    def get_inputs(self):
        return [
            paddle.uniform([5522, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([5522, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_8553dcbbbaea0506096effe6a6dbd925(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_791985e0863360b3b07d19d88c268b58
    def get_inputs(self):
        return [
            paddle.uniform([5522, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([5522, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_8553dcbbbaea0506096effe6a6dbd925(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_791985e0863360b3b07d19d88c268b58
    def get_inputs(self):
        return [
            paddle.uniform([5522, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([5522, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_9ac2539e1bfebdbbdaf888ff0d4461d3(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.maximum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None], dtype='float32'),
            paddle.static.InputSpec(shape=[None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_61c3d29099f43273eab3cd39428efab8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9ac2539e1bfebdbbdaf888ff0d4461d3
    def get_inputs(self):
        return [
            paddle.to_tensor([0.3605007231235504, 0.3333803117275238, 0.23454944789409637, 0.26296818256378174, 0.3854544162750244, 0.2304389327764511], dtype='float32').reshape([6]),
            paddle.to_tensor([0.4128625690937042, 0.3836251497268677, 0.3297223746776581, 0.1599111407995224, 0.4577946364879608, 0.3183137774467468], dtype='float32').reshape([6]),
        ]


class TestPrimitiveOp_b7f117251915ca1f7259d2735a7bb509(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9ac2539e1bfebdbbdaf888ff0d4461d3
    def get_inputs(self):
        return [
            paddle.to_tensor([0.42629650235176086, 0.18972466886043549, 0.20663975179195404, 0.15081170201301575, 0.4174017608165741, 0.3629100024700165], dtype='float32').reshape([6]),
            paddle.to_tensor([0.023622216656804085, 0.2635997235774994, 0.20144450664520264, 0.25364255905151367, 0.04704642668366432, 0.07614580541849136], dtype='float32').reshape([6]),
        ]


class TestPrimitiveOp_e1907e5765152a2beeb3a0a134f4b5ec(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9ac2539e1bfebdbbdaf888ff0d4461d3
    def get_inputs(self):
        return [
            paddle.to_tensor([0.3605007231235504, 0.3333803117275238, 0.23454944789409637, 0.26296818256378174, 0.3854544162750244, 0.2304389327764511], dtype='float32').reshape([6]),
            paddle.to_tensor([0.4335317313671112, 0.11488522589206696, 0.34194254875183105, 0.16626453399658203, 0.2805379629135132, 0.4297538995742798], dtype='float32').reshape([6]),
        ]


class TestPrimitiveOp_e6d3396c2ac31801014e46bce4c6dc41(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9ac2539e1bfebdbbdaf888ff0d4461d3
    def get_inputs(self):
        return [
            paddle.to_tensor([0.42629650235176086, 0.18972466886043549, 0.20663975179195404, 0.15081170201301575, 0.4174017608165741, 0.3629100024700165], dtype='float32').reshape([6]),
            paddle.to_tensor([0.013391555286943913, 0.3437585234642029, 0.4915377199649811, 0.19970282912254333, 0.21376128494739532, 0.1659878045320511], dtype='float32').reshape([6]),
        ]


class TestPrimitiveOp_c391a77a1b4d4eb8582dd7c3d3e0059f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9ac2539e1bfebdbbdaf888ff0d4461d3
    def get_inputs(self):
        return [
            paddle.to_tensor([0.4128625690937042, 0.3836251497268677, 0.3297223746776581, 0.26296818256378174, 0.4577946364879608, 0.3183137774467468], dtype='float32').reshape([6]),
            paddle.to_tensor([0.0922485888004303, 0.42701223492622375, 0.3167136311531067, 0.34751439094543457, 0.4444928765296936, 0.42101287841796875], dtype='float32').reshape([6]),
        ]


class TestPrimitiveOp_b0a74529d7a8df6b376b15f4dfc1712d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9ac2539e1bfebdbbdaf888ff0d4461d3
    def get_inputs(self):
        return [
            paddle.to_tensor([0.42629650235176086, 0.2635997235774994, 0.20663975179195404, 0.25364255905151367, 0.4174017608165741, 0.3629100024700165], dtype='float32').reshape([6]),
            paddle.to_tensor([0.1620912402868271, 0.20841006934642792, 0.06403285264968872, 0.4416305720806122, 0.15589286386966705, 0.45514115691185], dtype='float32').reshape([6]),
        ]


class TestPrimitiveOp_26a90679835044808fe12e6a9a8f8710(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_791985e0863360b3b07d19d88c268b58
    def get_inputs(self):
        return [
            paddle.uniform([1760, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1760, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_26a90679835044808fe12e6a9a8f8710(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_791985e0863360b3b07d19d88c268b58
    def get_inputs(self):
        return [
            paddle.uniform([1760, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1760, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_26a90679835044808fe12e6a9a8f8710(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_791985e0863360b3b07d19d88c268b58
    def get_inputs(self):
        return [
            paddle.uniform([1760, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1760, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_26a90679835044808fe12e6a9a8f8710(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_791985e0863360b3b07d19d88c268b58
    def get_inputs(self):
        return [
            paddle.uniform([1760, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1760, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_f3ceaaba5e2b0b3c8eb3086f8676ade3(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.maximum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_bfca07881ff803a36f86cb826aa87f85(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f3ceaaba5e2b0b3c8eb3086f8676ade3
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 38, 38], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.000000013351432e-10], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_2dc88c738ba8f4028e5c341aa7cf47f9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f00b14e252dc77869bbcd461a640cec3
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_2dc88c738ba8f4028e5c341aa7cf47f9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f00b14e252dc77869bbcd461a640cec3
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_32a2824b5753466216caf7b3a1b1fd89(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f00b14e252dc77869bbcd461a640cec3
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_32a2824b5753466216caf7b3a1b1fd89(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f00b14e252dc77869bbcd461a640cec3
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_eff8c18068440771efaedd141309a033(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_791985e0863360b3b07d19d88c268b58
    def get_inputs(self):
        return [
            paddle.uniform([1522, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1522, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_eff8c18068440771efaedd141309a033(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_791985e0863360b3b07d19d88c268b58
    def get_inputs(self):
        return [
            paddle.uniform([1522, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1522, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_eff8c18068440771efaedd141309a033(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_791985e0863360b3b07d19d88c268b58
    def get_inputs(self):
        return [
            paddle.uniform([1522, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1522, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_eff8c18068440771efaedd141309a033(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_791985e0863360b3b07d19d88c268b58
    def get_inputs(self):
        return [
            paddle.uniform([1522, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1522, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_1eb7a23dd374710aeca7d92950a807b0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f00b14e252dc77869bbcd461a640cec3
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_1eb7a23dd374710aeca7d92950a807b0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f00b14e252dc77869bbcd461a640cec3
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_bf2929b72dc2c817af734c7811bb283d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f00b14e252dc77869bbcd461a640cec3
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_bf2929b72dc2c817af734c7811bb283d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f00b14e252dc77869bbcd461a640cec3
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ac3c862390900f2f336e8f8fcaaddb76(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_791985e0863360b3b07d19d88c268b58
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.0983637347817421]], dtype='float32').reshape([1, 1]),
            paddle.to_tensor([[0.1288921982049942]], dtype='float32').reshape([1, 1]),
        ]


class TestPrimitiveOp_a661cfc9390008c5d46358af332ccccc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_791985e0863360b3b07d19d88c268b58
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.2091304212808609]], dtype='float32').reshape([1, 1]),
            paddle.to_tensor([[0.3986872434616089]], dtype='float32').reshape([1, 1]),
        ]


class TestPrimitiveOp_56ba92dcce26139daea927b9c727ff74(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_791985e0863360b3b07d19d88c268b58
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.06771773844957352]], dtype='float32').reshape([1, 1]),
            paddle.to_tensor([[0.11078812927007675]], dtype='float32').reshape([1, 1]),
        ]


class TestPrimitiveOp_8d030a2c3acae130d8a86a4b0366f19a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_791985e0863360b3b07d19d88c268b58
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.15742851793766022]], dtype='float32').reshape([1, 1]),
            paddle.to_tensor([[0.15218256413936615]], dtype='float32').reshape([1, 1]),
        ]


class TestPrimitiveOp_a65d0627b71a2b93b796bbe4b962346b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_791985e0863360b3b07d19d88c268b58
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.22831673920154572], [0.33949005603790283], [0.2869625687599182], [0.4246893525123596], [0.002240711124613881], [0.14619268476963043]], dtype='float32').reshape([6, 1]),
            paddle.to_tensor([[0.015245765447616577], [0.23714831471443176], [0.2166336476802826], [0.3967365324497223], [0.16642236709594727], [0.31202274560928345]], dtype='float32').reshape([6, 1]),
        ]


class TestPrimitiveOp_3b1c240fff5c55cc58d06ed3c545e197(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_791985e0863360b3b07d19d88c268b58
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.1350128948688507], [0.47312918305397034], [0.32580894231796265], [0.46865251660346985], [0.4675268828868866], [0.08211576193571091]], dtype='float32').reshape([6, 1]),
            paddle.to_tensor([[0.1296968311071396], [0.11471404135227203], [0.3752787411212921], [0.31934216618537903], [0.48315006494522095], [0.45246943831443787]], dtype='float32').reshape([6, 1]),
        ]


class TestPrimitiveOp_5831227a21cd9ac2c0c029f8d912bbd0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_791985e0863360b3b07d19d88c268b58
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.24626027047634125], [0.14295130968093872], [0.4793103039264679], [0.25223153829574585], [0.03669232502579689], [0.17557862401008606]], dtype='float32').reshape([6, 1]),
            paddle.to_tensor([[0.22387434542179108], [0.044702861458063126], [0.20448708534240723], [0.24163779616355896], [0.2939894199371338], [0.43704453110694885]], dtype='float32').reshape([6, 1]),
        ]


class TestPrimitiveOp_be7b73b184f13511b3d74e5f6d36f665(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_791985e0863360b3b07d19d88c268b58
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.20023022592067719], [0.2667973041534424], [0.011914309114217758], [0.18980513513088226], [0.43041738867759705], [0.09507034718990326]], dtype='float32').reshape([6, 1]),
            paddle.to_tensor([[0.4093327224254608], [0.3914657533168793], [0.4158235192298889], [0.16640233993530273], [0.04521951824426651], [0.3970659077167511]], dtype='float32').reshape([6, 1]),
        ]


class TestPrimitiveOp_5fab66a1508f07ff510e57a7abf0e931(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f00b14e252dc77869bbcd461a640cec3
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_5fab66a1508f07ff510e57a7abf0e931(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f00b14e252dc77869bbcd461a640cec3
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_47251a23ac117016f7e358b0d1d45e19(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f00b14e252dc77869bbcd461a640cec3
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_47251a23ac117016f7e358b0d1d45e19(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f00b14e252dc77869bbcd461a640cec3
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ce93e9cce0ae9614b269eea2e406c8eb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_791985e0863360b3b07d19d88c268b58
    def get_inputs(self):
        return [
            paddle.uniform([2074, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2074, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ce93e9cce0ae9614b269eea2e406c8eb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_791985e0863360b3b07d19d88c268b58
    def get_inputs(self):
        return [
            paddle.uniform([2074, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2074, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ce93e9cce0ae9614b269eea2e406c8eb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_791985e0863360b3b07d19d88c268b58
    def get_inputs(self):
        return [
            paddle.uniform([2074, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2074, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ce93e9cce0ae9614b269eea2e406c8eb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_791985e0863360b3b07d19d88c268b58
    def get_inputs(self):
        return [
            paddle.uniform([2074, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2074, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_60889c0bc9566c453f1332eba9df7d0d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f00b14e252dc77869bbcd461a640cec3
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_60889c0bc9566c453f1332eba9df7d0d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f00b14e252dc77869bbcd461a640cec3
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ffaeb74279fa58e686ca81de16652e8b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_791985e0863360b3b07d19d88c268b58
    def get_inputs(self):
        return [
            paddle.uniform([4734, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4734, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ffaeb74279fa58e686ca81de16652e8b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_791985e0863360b3b07d19d88c268b58
    def get_inputs(self):
        return [
            paddle.uniform([4734, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4734, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ffaeb74279fa58e686ca81de16652e8b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_791985e0863360b3b07d19d88c268b58
    def get_inputs(self):
        return [
            paddle.uniform([4734, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4734, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ffaeb74279fa58e686ca81de16652e8b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_791985e0863360b3b07d19d88c268b58
    def get_inputs(self):
        return [
            paddle.uniform([4734, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4734, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_15c8a06e30473da39d12cff613bd6bfb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_791985e0863360b3b07d19d88c268b58
    def get_inputs(self):
        return [
            paddle.uniform([1074, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1074, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_15c8a06e30473da39d12cff613bd6bfb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_791985e0863360b3b07d19d88c268b58
    def get_inputs(self):
        return [
            paddle.uniform([1074, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1074, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_15c8a06e30473da39d12cff613bd6bfb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_791985e0863360b3b07d19d88c268b58
    def get_inputs(self):
        return [
            paddle.uniform([1074, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1074, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_15c8a06e30473da39d12cff613bd6bfb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_791985e0863360b3b07d19d88c268b58
    def get_inputs(self):
        return [
            paddle.uniform([1074, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1074, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c92cbf8524e253b9523a281d87c603a6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f00b14e252dc77869bbcd461a640cec3
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c92cbf8524e253b9523a281d87c603a6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f00b14e252dc77869bbcd461a640cec3
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e0aa7dbeb1efb983f626189bbc76d586(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f00b14e252dc77869bbcd461a640cec3
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e0aa7dbeb1efb983f626189bbc76d586(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f00b14e252dc77869bbcd461a640cec3
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_1d76c4b314d549bcae1652c375cec333(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_791985e0863360b3b07d19d88c268b58
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.17697207629680634], [0.04053914546966553], [0.00564520712941885], [0.05893722176551819], [0.38709262013435364]], dtype='float32').reshape([5, 1]),
            paddle.to_tensor([[0.0247951690107584], [0.29903921484947205], [0.13782429695129395], [0.46026167273521423], [0.1342550814151764]], dtype='float32').reshape([5, 1]),
        ]


class TestPrimitiveOp_89aa7a1a6cdee2de9c6f0e04166dfb4c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_791985e0863360b3b07d19d88c268b58
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.39805009961128235], [0.3606886863708496], [0.03755515068769455], [0.2986750900745392], [0.32391923666000366]], dtype='float32').reshape([5, 1]),
            paddle.to_tensor([[0.08429738134145737], [0.4595024883747101], [0.12538257241249084], [0.4333687126636505], [0.47472846508026123]], dtype='float32').reshape([5, 1]),
        ]


class TestPrimitiveOp_bdbb76db48cfd7cbd5399f5d10980594(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_791985e0863360b3b07d19d88c268b58
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.10796797275543213], [0.4512024223804474], [0.11816670745611191], [0.465546578168869], [0.21543511748313904]], dtype='float32').reshape([5, 1]),
            paddle.to_tensor([[0.42674073576927185], [0.07695849984884262], [0.3584158718585968], [0.017290905117988586], [0.11618880927562714]], dtype='float32').reshape([5, 1]),
        ]


class TestPrimitiveOp_e22b06ef52c29818b5636a38e8306b4a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_791985e0863360b3b07d19d88c268b58
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.15855374932289124], [0.13624975085258484], [0.12530755996704102], [0.2953103184700012], [0.01684260368347168]], dtype='float32').reshape([5, 1]),
            paddle.to_tensor([[0.28606072068214417], [0.3674119710922241], [0.05144762992858887], [0.40608474612236023], [0.32032695412635803]], dtype='float32').reshape([5, 1]),
        ]


class TestPrimitiveOp_8dd6488a8eb6a91ac7fd0f52405ef2df(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f00b14e252dc77869bbcd461a640cec3
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_8dd6488a8eb6a91ac7fd0f52405ef2df(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f00b14e252dc77869bbcd461a640cec3
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a20e0871324731e58726ec0dcc7e339d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f00b14e252dc77869bbcd461a640cec3
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a20e0871324731e58726ec0dcc7e339d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f00b14e252dc77869bbcd461a640cec3
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b0bfc52163a4a1968cf92f8e8276cee2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f00b14e252dc77869bbcd461a640cec3
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b0bfc52163a4a1968cf92f8e8276cee2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f00b14e252dc77869bbcd461a640cec3
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_24c51aaa127a5266d18b8f49da2377ad(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_791985e0863360b3b07d19d88c268b58
    def get_inputs(self):
        return [
            paddle.uniform([2332, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2332, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_24c51aaa127a5266d18b8f49da2377ad(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_791985e0863360b3b07d19d88c268b58
    def get_inputs(self):
        return [
            paddle.uniform([2332, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2332, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_24c51aaa127a5266d18b8f49da2377ad(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_791985e0863360b3b07d19d88c268b58
    def get_inputs(self):
        return [
            paddle.uniform([2332, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2332, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_24c51aaa127a5266d18b8f49da2377ad(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_791985e0863360b3b07d19d88c268b58
    def get_inputs(self):
        return [
            paddle.uniform([2332, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2332, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_49057f187fbab41ad634eecca6434230(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_791985e0863360b3b07d19d88c268b58
    def get_inputs(self):
        return [
            paddle.uniform([3051, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3051, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_49057f187fbab41ad634eecca6434230(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_791985e0863360b3b07d19d88c268b58
    def get_inputs(self):
        return [
            paddle.uniform([3051, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3051, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_49057f187fbab41ad634eecca6434230(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_791985e0863360b3b07d19d88c268b58
    def get_inputs(self):
        return [
            paddle.uniform([3051, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3051, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_49057f187fbab41ad634eecca6434230(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_791985e0863360b3b07d19d88c268b58
    def get_inputs(self):
        return [
            paddle.uniform([3051, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3051, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_5d4082c0b726dbc4287b18bc105a32c4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_791985e0863360b3b07d19d88c268b58
    def get_inputs(self):
        return [
            paddle.uniform([3870, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3870, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_5d4082c0b726dbc4287b18bc105a32c4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_791985e0863360b3b07d19d88c268b58
    def get_inputs(self):
        return [
            paddle.uniform([3870, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3870, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_5d4082c0b726dbc4287b18bc105a32c4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_791985e0863360b3b07d19d88c268b58
    def get_inputs(self):
        return [
            paddle.uniform([3870, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3870, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_5d4082c0b726dbc4287b18bc105a32c4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_791985e0863360b3b07d19d88c268b58
    def get_inputs(self):
        return [
            paddle.uniform([3870, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3870, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_60889c0bc9566c453f1332eba9df7d0d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f00b14e252dc77869bbcd461a640cec3
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_60889c0bc9566c453f1332eba9df7d0d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f00b14e252dc77869bbcd461a640cec3
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_48d2ca5e09ff6cc545156a409573aaa5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_791985e0863360b3b07d19d88c268b58
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.4567643404006958], [0.008849719539284706], [0.003253927920013666], [0.2674586772918701]], dtype='float32').reshape([4, 1]),
            paddle.to_tensor([[0.3858823776245117], [0.4713914096355438], [0.25339508056640625], [0.21384595334529877]], dtype='float32').reshape([4, 1]),
        ]


class TestPrimitiveOp_0863685b07af967ec764825a9f863f5f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_791985e0863360b3b07d19d88c268b58
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.11311373114585876], [0.26892995834350586], [0.4244452118873596], [0.09755375981330872]], dtype='float32').reshape([4, 1]),
            paddle.to_tensor([[0.2055460810661316], [0.16424299776554108], [0.13470056653022766], [0.07356300950050354]], dtype='float32').reshape([4, 1]),
        ]


class TestPrimitiveOp_719f7522340ef275bf17f640fe44b090(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_791985e0863360b3b07d19d88c268b58
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.006803394760936499], [0.40538290143013], [0.19567006826400757], [0.4959845542907715]], dtype='float32').reshape([4, 1]),
            paddle.to_tensor([[0.18735556304454803], [0.1294925957918167], [0.19429393112659454], [0.0008755988092161715]], dtype='float32').reshape([4, 1]),
        ]


class TestPrimitiveOp_c338f03d008ac8a84425910066ae8e22(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_791985e0863360b3b07d19d88c268b58
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.21126195788383484], [0.052910272032022476], [0.37054139375686646], [0.038458243012428284]], dtype='float32').reshape([4, 1]),
            paddle.to_tensor([[0.3032015860080719], [0.3202653229236603], [0.1812242865562439], [0.20975439250469208]], dtype='float32').reshape([4, 1]),
        ]


class TestPrimitiveOp_83e3914605b142c74d9f40832b458445(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_791985e0863360b3b07d19d88c268b58
    def get_inputs(self):
        return [
            paddle.uniform([2111, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2111, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_83e3914605b142c74d9f40832b458445(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_791985e0863360b3b07d19d88c268b58
    def get_inputs(self):
        return [
            paddle.uniform([2111, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2111, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_83e3914605b142c74d9f40832b458445(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_791985e0863360b3b07d19d88c268b58
    def get_inputs(self):
        return [
            paddle.uniform([2111, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2111, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_83e3914605b142c74d9f40832b458445(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_791985e0863360b3b07d19d88c268b58
    def get_inputs(self):
        return [
            paddle.uniform([2111, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2111, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b0bfc52163a4a1968cf92f8e8276cee2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f00b14e252dc77869bbcd461a640cec3
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b0bfc52163a4a1968cf92f8e8276cee2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f00b14e252dc77869bbcd461a640cec3
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3f3dc7b350f9ad1862de696709cb6b25(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f00b14e252dc77869bbcd461a640cec3
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3f3dc7b350f9ad1862de696709cb6b25(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f00b14e252dc77869bbcd461a640cec3
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a246e68cd5fc520aa2972134b2892762(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f00b14e252dc77869bbcd461a640cec3
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a246e68cd5fc520aa2972134b2892762(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f00b14e252dc77869bbcd461a640cec3
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_0739f5e2b46d2e97a9cceba1ca2e050d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_791985e0863360b3b07d19d88c268b58
    def get_inputs(self):
        return [
            paddle.uniform([4141, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4141, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_0739f5e2b46d2e97a9cceba1ca2e050d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_791985e0863360b3b07d19d88c268b58
    def get_inputs(self):
        return [
            paddle.uniform([4141, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4141, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_0739f5e2b46d2e97a9cceba1ca2e050d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_791985e0863360b3b07d19d88c268b58
    def get_inputs(self):
        return [
            paddle.uniform([4141, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4141, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_0739f5e2b46d2e97a9cceba1ca2e050d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_791985e0863360b3b07d19d88c268b58
    def get_inputs(self):
        return [
            paddle.uniform([4141, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4141, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a246e68cd5fc520aa2972134b2892762(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f00b14e252dc77869bbcd461a640cec3
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a246e68cd5fc520aa2972134b2892762(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f00b14e252dc77869bbcd461a640cec3
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
        ]




if __name__ == '__main__':
    unittest.main()