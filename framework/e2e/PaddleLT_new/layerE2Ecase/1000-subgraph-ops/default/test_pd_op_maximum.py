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


class PrimitiveOp_72553509889f68e7b8f7aca55694b8a0(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.maximum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_55bd71a33220525e44f4dc07d056b8f0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_72553509889f68e7b8f7aca55694b8a0
    def get_inputs(self):
        return [
            paddle.uniform([1756, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1756, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_55bd71a33220525e44f4dc07d056b8f0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_72553509889f68e7b8f7aca55694b8a0
    def get_inputs(self):
        return [
            paddle.uniform([1756, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1756, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_55bd71a33220525e44f4dc07d056b8f0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_72553509889f68e7b8f7aca55694b8a0
    def get_inputs(self):
        return [
            paddle.uniform([1756, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1756, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_55bd71a33220525e44f4dc07d056b8f0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_72553509889f68e7b8f7aca55694b8a0
    def get_inputs(self):
        return [
            paddle.uniform([1756, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1756, 1], dtype='float32', min=0, max=0.5),
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


class TestPrimitiveOp_5151cffb6ef0352699a718d7e2de661e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_791985e0863360b3b07d19d88c268b58
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.4907219707965851], [0.47655177116394043], [0.45146989822387695], [0.2179049253463745], [0.2145148366689682], [0.42308953404426575], [0.2811638414859772], [0.3096412420272827], [0.08616721630096436]], dtype='float32').reshape([9, 1]),
            paddle.to_tensor([[0.4125475287437439], [0.2618580162525177], [0.14666202664375305], [0.37152010202407837], [0.4038541316986084], [0.4664810597896576], [0.22648189961910248], [0.3452494740486145], [0.3210054337978363]], dtype='float32').reshape([9, 1]),
        ]


class TestPrimitiveOp_99ac2f61e4be68bd0e6b6e92f5fa32e3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_791985e0863360b3b07d19d88c268b58
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.4733383357524872], [0.09571289271116257], [0.15195058286190033], [0.012989894486963749], [0.4056450426578522], [0.43923839926719666], [0.357232004404068], [0.16211768984794617], [0.025704603642225266]], dtype='float32').reshape([9, 1]),
            paddle.to_tensor([[0.4564463198184967], [0.20966672897338867], [0.15613777935504913], [0.4914016127586365], [0.3907115161418915], [0.38277748227119446], [0.43668332695961], [0.4740495979785919], [0.0606311671435833]], dtype='float32').reshape([9, 1]),
        ]


class TestPrimitiveOp_91b092f4203c521bf58b80bfb8401d6e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_791985e0863360b3b07d19d88c268b58
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.3370451331138611], [0.025026828050613403], [0.4009562134742737], [0.37868067622184753], [0.15391626954078674], [0.050961192697286606], [0.3595850467681885], [0.23399347066879272], [0.21992023289203644]], dtype='float32').reshape([9, 1]),
            paddle.to_tensor([[0.34383171796798706], [0.27951961755752563], [0.04696405306458473], [0.1218668669462204], [0.46935001015663147], [0.4291514456272125], [0.30239981412887573], [0.36811432242393494], [0.2899019420146942]], dtype='float32').reshape([9, 1]),
        ]


class TestPrimitiveOp_50295f17bb9b35eac62432470ea52a98(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_791985e0863360b3b07d19d88c268b58
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.03724591061472893], [0.011372104287147522], [0.10467681288719177], [0.45974865555763245], [0.47817090153694153], [0.09304334223270416], [0.3312247097492218], [0.3817911446094513], [0.08097352087497711]], dtype='float32').reshape([9, 1]),
            paddle.to_tensor([[0.3704785704612732], [0.022056521847844124], [0.3112718164920807], [0.13299866020679474], [0.3006611168384552], [0.48700836300849915], [0.43817874789237976], [0.11871779710054398], [0.35156527161598206]], dtype='float32').reshape([9, 1]),
        ]


class TestPrimitiveOp_dd68d4eda91d10cae3c772d724e0ac3b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_72553509889f68e7b8f7aca55694b8a0
    def get_inputs(self):
        return [
            paddle.uniform([5551, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([5551, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_dd68d4eda91d10cae3c772d724e0ac3b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_72553509889f68e7b8f7aca55694b8a0
    def get_inputs(self):
        return [
            paddle.uniform([5551, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([5551, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_dd68d4eda91d10cae3c772d724e0ac3b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_72553509889f68e7b8f7aca55694b8a0
    def get_inputs(self):
        return [
            paddle.uniform([5551, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([5551, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_dd68d4eda91d10cae3c772d724e0ac3b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_72553509889f68e7b8f7aca55694b8a0
    def get_inputs(self):
        return [
            paddle.uniform([5551, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([5551, 1], dtype='float32', min=0, max=0.5),
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


class TestPrimitiveOp_e596a6438e3469f3bc75d086305a69f2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9ac2539e1bfebdbbdaf888ff0d4461d3
    def get_inputs(self):
        return [
            paddle.to_tensor([0.3997354805469513, 0.39295482635498047, 0.09432562440633774, 0.49924978613853455, 0.30787521600723267, 0.35715749859809875], dtype='float32').reshape([6]),
            paddle.to_tensor([0.4078967571258545, 0.37209582328796387, 0.23028191924095154, 0.16957210004329681, 0.18082909286022186, 0.4491502642631531], dtype='float32').reshape([6]),
        ]


class TestPrimitiveOp_9b21ec588fbcf84f13f3dbe9dbb8305c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9ac2539e1bfebdbbdaf888ff0d4461d3
    def get_inputs(self):
        return [
            paddle.to_tensor([0.4620789587497711, 0.2747343182563782, 0.18587720394134521, 0.3222280740737915, 0.2506195902824402, 0.2472265213727951], dtype='float32').reshape([6]),
            paddle.to_tensor([0.18946067988872528, 0.18211600184440613, 0.004157352261245251, 0.01153471041470766, 0.360180139541626, 0.3169853091239929], dtype='float32').reshape([6]),
        ]


class TestPrimitiveOp_f427664740c4b41818bb2da09feda3db(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9ac2539e1bfebdbbdaf888ff0d4461d3
    def get_inputs(self):
        return [
            paddle.to_tensor([0.3997354805469513, 0.39295482635498047, 0.09432562440633774, 0.49924978613853455, 0.30787521600723267, 0.35715749859809875], dtype='float32').reshape([6]),
            paddle.to_tensor([0.0959656834602356, 0.010401327162981033, 0.11267633736133575, 0.08521786332130432, 0.09726300835609436, 0.05800497531890869], dtype='float32').reshape([6]),
        ]


class TestPrimitiveOp_10f53a0844f9657147f6b338f05780c8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9ac2539e1bfebdbbdaf888ff0d4461d3
    def get_inputs(self):
        return [
            paddle.to_tensor([0.4620789587497711, 0.2747343182563782, 0.18587720394134521, 0.3222280740737915, 0.2506195902824402, 0.2472265213727951], dtype='float32').reshape([6]),
            paddle.to_tensor([0.08197572827339172, 0.33816057443618774, 0.29246312379837036, 0.4799637198448181, 0.3316073417663574, 0.26939353346824646], dtype='float32').reshape([6]),
        ]


class TestPrimitiveOp_0695fe90cd9510ec5731386ea45b4c5a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9ac2539e1bfebdbbdaf888ff0d4461d3
    def get_inputs(self):
        return [
            paddle.to_tensor([0.4078967571258545, 0.39295482635498047, 0.23028191924095154, 0.49924978613853455, 0.30787521600723267, 0.4491502642631531], dtype='float32').reshape([6]),
            paddle.to_tensor([0.056108973920345306, 0.36671507358551025, 0.13837383687496185, 0.014441891573369503, 0.34063658118247986, 0.30013731122016907], dtype='float32').reshape([6]),
        ]


class TestPrimitiveOp_18002075ab7dd70dc540be1407f96827(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9ac2539e1bfebdbbdaf888ff0d4461d3
    def get_inputs(self):
        return [
            paddle.to_tensor([0.4620789587497711, 0.2747343182563782, 0.18587720394134521, 0.3222280740737915, 0.360180139541626, 0.3169853091239929], dtype='float32').reshape([6]),
            paddle.to_tensor([0.18877169489860535, 0.04359738901257515, 0.3416917026042938, 0.2778036594390869, 0.22354553639888763, 0.17368389666080475], dtype='float32').reshape([6]),
        ]


class TestPrimitiveOp_71613c5592eb94d5c09b4d08a8009b22(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_72553509889f68e7b8f7aca55694b8a0
    def get_inputs(self):
        return [
            paddle.uniform([1769, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1769, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_71613c5592eb94d5c09b4d08a8009b22(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_72553509889f68e7b8f7aca55694b8a0
    def get_inputs(self):
        return [
            paddle.uniform([1769, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1769, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_71613c5592eb94d5c09b4d08a8009b22(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_72553509889f68e7b8f7aca55694b8a0
    def get_inputs(self):
        return [
            paddle.uniform([1769, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1769, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_71613c5592eb94d5c09b4d08a8009b22(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_72553509889f68e7b8f7aca55694b8a0
    def get_inputs(self):
        return [
            paddle.uniform([1769, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1769, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_b988090c8a8e9bd8d7e5120a78832804(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.maximum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 1, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_0ecf30f4b7e61d52daa5219030463807(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b988090c8a8e9bd8d7e5120a78832804
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


class TestPrimitiveOp_b7d8558fcde08c31a41a1e6fbf0535d0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_72553509889f68e7b8f7aca55694b8a0
    def get_inputs(self):
        return [
            paddle.uniform([1502, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1502, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b7d8558fcde08c31a41a1e6fbf0535d0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_72553509889f68e7b8f7aca55694b8a0
    def get_inputs(self):
        return [
            paddle.uniform([1502, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1502, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b7d8558fcde08c31a41a1e6fbf0535d0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_72553509889f68e7b8f7aca55694b8a0
    def get_inputs(self):
        return [
            paddle.uniform([1502, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1502, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b7d8558fcde08c31a41a1e6fbf0535d0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_72553509889f68e7b8f7aca55694b8a0
    def get_inputs(self):
        return [
            paddle.uniform([1502, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1502, 1], dtype='float32', min=0, max=0.5),
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


class TestPrimitiveOp_d68c8c921fa4a9b32c9adc2e4e846d4f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_791985e0863360b3b07d19d88c268b58
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.15802021324634552]], dtype='float32').reshape([1, 1]),
            paddle.to_tensor([[0.028114398941397667]], dtype='float32').reshape([1, 1]),
        ]


class TestPrimitiveOp_3e8ea306b02426d0190de5eefc137f60(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_791985e0863360b3b07d19d88c268b58
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.288260281085968]], dtype='float32').reshape([1, 1]),
            paddle.to_tensor([[0.2992727756500244]], dtype='float32').reshape([1, 1]),
        ]


class TestPrimitiveOp_5fa64be15483d36757fc23afe079c93c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_791985e0863360b3b07d19d88c268b58
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.3475726544857025]], dtype='float32').reshape([1, 1]),
            paddle.to_tensor([[0.1071244403719902]], dtype='float32').reshape([1, 1]),
        ]


class TestPrimitiveOp_a3d538822e9ca466cd50ef12ce703d5b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_791985e0863360b3b07d19d88c268b58
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.47082066535949707]], dtype='float32').reshape([1, 1]),
            paddle.to_tensor([[0.32788488268852234]], dtype='float32').reshape([1, 1]),
        ]


class TestPrimitiveOp_841070a790e3627bc4775d677c32bc7c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_791985e0863360b3b07d19d88c268b58
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.41157740354537964], [0.4826880991458893], [0.4846378266811371], [0.25960448384284973], [0.1073368713259697], [0.08710668981075287]], dtype='float32').reshape([6, 1]),
            paddle.to_tensor([[0.3591238260269165], [0.3641086220741272], [0.1448068916797638], [0.0998174175620079], [0.16274337470531464], [0.04630977287888527]], dtype='float32').reshape([6, 1]),
        ]


class TestPrimitiveOp_a371aaa5719350de778cc6307c5a984c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_791985e0863360b3b07d19d88c268b58
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.4798603653907776], [0.32325857877731323], [0.02578026056289673], [0.1680978387594223], [0.018725527450442314], [0.12951743602752686]], dtype='float32').reshape([6, 1]),
            paddle.to_tensor([[0.22245408594608307], [0.08651915937662125], [0.2651921510696411], [0.07728420943021774], [0.03530534356832504], [0.18742431700229645]], dtype='float32').reshape([6, 1]),
        ]


class TestPrimitiveOp_335019066ee5a419ac17c2544f49d3f0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_791985e0863360b3b07d19d88c268b58
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.19522464275360107], [0.29033544659614563], [0.25756993889808655], [0.4750064015388489], [0.4037943184375763], [0.015598968602716923]], dtype='float32').reshape([6, 1]),
            paddle.to_tensor([[0.2858388125896454], [0.3007555902004242], [0.3974524438381195], [0.002446115715429187], [0.14687369763851166], [0.48988133668899536]], dtype='float32').reshape([6, 1]),
        ]


class TestPrimitiveOp_9830cb6a920398b710d616e32995c217(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_791985e0863360b3b07d19d88c268b58
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.498967707157135], [0.11776749044656754], [0.10755512863397598], [0.23614223301410675], [0.1757691651582718], [0.31581953167915344]], dtype='float32').reshape([6, 1]),
            paddle.to_tensor([[0.016921646893024445], [0.35783061385154724], [0.12393762171268463], [0.2560504674911499], [0.055858783423900604], [0.2302117496728897]], dtype='float32').reshape([6, 1]),
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


class TestPrimitiveOp_91ca0f264dacfb713581cf4b64301bca(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_72553509889f68e7b8f7aca55694b8a0
    def get_inputs(self):
        return [
            paddle.uniform([2080, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2080, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_91ca0f264dacfb713581cf4b64301bca(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_72553509889f68e7b8f7aca55694b8a0
    def get_inputs(self):
        return [
            paddle.uniform([2080, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2080, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_91ca0f264dacfb713581cf4b64301bca(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_72553509889f68e7b8f7aca55694b8a0
    def get_inputs(self):
        return [
            paddle.uniform([2080, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2080, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_91ca0f264dacfb713581cf4b64301bca(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_72553509889f68e7b8f7aca55694b8a0
    def get_inputs(self):
        return [
            paddle.uniform([2080, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2080, 1], dtype='float32', min=0, max=0.5),
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


class TestPrimitiveOp_c5ad3a98fc765cf731bff820907c873d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_72553509889f68e7b8f7aca55694b8a0
    def get_inputs(self):
        return [
            paddle.uniform([4585, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4585, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c5ad3a98fc765cf731bff820907c873d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_72553509889f68e7b8f7aca55694b8a0
    def get_inputs(self):
        return [
            paddle.uniform([4585, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4585, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c5ad3a98fc765cf731bff820907c873d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_72553509889f68e7b8f7aca55694b8a0
    def get_inputs(self):
        return [
            paddle.uniform([4585, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4585, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c5ad3a98fc765cf731bff820907c873d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_72553509889f68e7b8f7aca55694b8a0
    def get_inputs(self):
        return [
            paddle.uniform([4585, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4585, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c3a74bb2949611878c1eb96088ef8c8a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_72553509889f68e7b8f7aca55694b8a0
    def get_inputs(self):
        return [
            paddle.uniform([1048, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1048, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c3a74bb2949611878c1eb96088ef8c8a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_72553509889f68e7b8f7aca55694b8a0
    def get_inputs(self):
        return [
            paddle.uniform([1048, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1048, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c3a74bb2949611878c1eb96088ef8c8a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_72553509889f68e7b8f7aca55694b8a0
    def get_inputs(self):
        return [
            paddle.uniform([1048, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1048, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c3a74bb2949611878c1eb96088ef8c8a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_72553509889f68e7b8f7aca55694b8a0
    def get_inputs(self):
        return [
            paddle.uniform([1048, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1048, 1], dtype='float32', min=0, max=0.5),
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


class TestPrimitiveOp_f0fa8dd1883b1b33765ee0f952b98a93(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_791985e0863360b3b07d19d88c268b58
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.19752667844295502], [0.10475891828536987], [0.024959586560726166], [0.056447286158800125], [0.1750042885541916]], dtype='float32').reshape([5, 1]),
            paddle.to_tensor([[0.37055835127830505], [0.19263240694999695], [0.2666272819042206], [0.045508455485105515], [0.30655404925346375]], dtype='float32').reshape([5, 1]),
        ]


class TestPrimitiveOp_fdc5a1df697238c9f7c775513bccbc2c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_791985e0863360b3b07d19d88c268b58
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.07073542475700378], [0.2629903554916382], [0.18045572936534882], [0.3250402510166168], [0.24727387726306915]], dtype='float32').reshape([5, 1]),
            paddle.to_tensor([[0.29983288049697876], [0.19387735426425934], [0.1999022364616394], [0.3063305914402008], [0.06010952964425087]], dtype='float32').reshape([5, 1]),
        ]


class TestPrimitiveOp_ce1583d366d164adf636e384540fc696(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_791985e0863360b3b07d19d88c268b58
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.48249489068984985], [0.19764673709869385], [0.020277973264455795], [0.3286098837852478], [0.24283719062805176]], dtype='float32').reshape([5, 1]),
            paddle.to_tensor([[0.13977095484733582], [0.2654637098312378], [0.3343604803085327], [0.33652734756469727], [0.13251619040966034]], dtype='float32').reshape([5, 1]),
        ]


class TestPrimitiveOp_0505884db16ea24b985b7e5266d2751b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_791985e0863360b3b07d19d88c268b58
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.47020426392555237], [0.4614999294281006], [0.42352351546287537], [0.24572992324829102], [0.3263440430164337]], dtype='float32').reshape([5, 1]),
            paddle.to_tensor([[0.025554388761520386], [0.1028725877404213], [0.3705821931362152], [0.4677906334400177], [0.2983556091785431]], dtype='float32').reshape([5, 1]),
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


class TestPrimitiveOp_65d1ad2602c6610a1242702f92f68e98(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_72553509889f68e7b8f7aca55694b8a0
    def get_inputs(self):
        return [
            paddle.uniform([2390, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2390, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_65d1ad2602c6610a1242702f92f68e98(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_72553509889f68e7b8f7aca55694b8a0
    def get_inputs(self):
        return [
            paddle.uniform([2390, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2390, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_65d1ad2602c6610a1242702f92f68e98(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_72553509889f68e7b8f7aca55694b8a0
    def get_inputs(self):
        return [
            paddle.uniform([2390, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2390, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_65d1ad2602c6610a1242702f92f68e98(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_72553509889f68e7b8f7aca55694b8a0
    def get_inputs(self):
        return [
            paddle.uniform([2390, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2390, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f7ccdc6983ee2b47db6901635656a0c1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_72553509889f68e7b8f7aca55694b8a0
    def get_inputs(self):
        return [
            paddle.uniform([3090, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3090, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f7ccdc6983ee2b47db6901635656a0c1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_72553509889f68e7b8f7aca55694b8a0
    def get_inputs(self):
        return [
            paddle.uniform([3090, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3090, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f7ccdc6983ee2b47db6901635656a0c1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_72553509889f68e7b8f7aca55694b8a0
    def get_inputs(self):
        return [
            paddle.uniform([3090, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3090, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f7ccdc6983ee2b47db6901635656a0c1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_72553509889f68e7b8f7aca55694b8a0
    def get_inputs(self):
        return [
            paddle.uniform([3090, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3090, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_0d55f86deefb98a72ecb4638c45c5889(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_72553509889f68e7b8f7aca55694b8a0
    def get_inputs(self):
        return [
            paddle.uniform([3748, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3748, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_0d55f86deefb98a72ecb4638c45c5889(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_72553509889f68e7b8f7aca55694b8a0
    def get_inputs(self):
        return [
            paddle.uniform([3748, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3748, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_0d55f86deefb98a72ecb4638c45c5889(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_72553509889f68e7b8f7aca55694b8a0
    def get_inputs(self):
        return [
            paddle.uniform([3748, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3748, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_0d55f86deefb98a72ecb4638c45c5889(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_72553509889f68e7b8f7aca55694b8a0
    def get_inputs(self):
        return [
            paddle.uniform([3748, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3748, 1], dtype='float32', min=0, max=0.5),
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


class TestPrimitiveOp_a6055fad2dbfb370b81730b08a3c8a6b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_791985e0863360b3b07d19d88c268b58
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.37155959010124207], [0.37738490104675293], [0.38180166482925415], [0.16988332569599152]], dtype='float32').reshape([4, 1]),
            paddle.to_tensor([[0.37256571650505066], [0.25770851969718933], [0.40263471007347107], [0.15933819115161896]], dtype='float32').reshape([4, 1]),
        ]


class TestPrimitiveOp_38584fecccaf9a6db756ac1bf5e8a90d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_791985e0863360b3b07d19d88c268b58
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.35161545872688293], [0.1566532999277115], [0.24491369724273682], [0.47443392872810364]], dtype='float32').reshape([4, 1]),
            paddle.to_tensor([[0.4308514893054962], [0.21365579962730408], [0.3567523658275604], [0.20909146964550018]], dtype='float32').reshape([4, 1]),
        ]


class TestPrimitiveOp_640d1c407ced3975c672ad9c5867c2e9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_791985e0863360b3b07d19d88c268b58
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.006397643592208624], [0.22528007626533508], [0.4595484733581543], [0.404490202665329]], dtype='float32').reshape([4, 1]),
            paddle.to_tensor([[0.3430819809436798], [0.16135065257549286], [0.05680867284536362], [0.19531609117984772]], dtype='float32').reshape([4, 1]),
        ]


class TestPrimitiveOp_4fc5fe93eb0a0c21a399031778a892a7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_791985e0863360b3b07d19d88c268b58
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.15801455080509186], [0.03819771856069565], [0.08493614196777344], [0.2796372175216675]], dtype='float32').reshape([4, 1]),
            paddle.to_tensor([[0.097120501101017], [0.3680274784564972], [0.0860719233751297], [0.2551638185977936]], dtype='float32').reshape([4, 1]),
        ]


class TestPrimitiveOp_e29c6b6503f2d10ffcf53f89e69b2f1f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_72553509889f68e7b8f7aca55694b8a0
    def get_inputs(self):
        return [
            paddle.uniform([2031, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2031, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e29c6b6503f2d10ffcf53f89e69b2f1f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_72553509889f68e7b8f7aca55694b8a0
    def get_inputs(self):
        return [
            paddle.uniform([2031, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2031, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e29c6b6503f2d10ffcf53f89e69b2f1f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_72553509889f68e7b8f7aca55694b8a0
    def get_inputs(self):
        return [
            paddle.uniform([2031, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2031, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e29c6b6503f2d10ffcf53f89e69b2f1f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_72553509889f68e7b8f7aca55694b8a0
    def get_inputs(self):
        return [
            paddle.uniform([2031, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2031, 1], dtype='float32', min=0, max=0.5),
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


class TestPrimitiveOp_aa323162a0a56599e198e83bd362f640(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_72553509889f68e7b8f7aca55694b8a0
    def get_inputs(self):
        return [
            paddle.uniform([4205, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4205, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_aa323162a0a56599e198e83bd362f640(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_72553509889f68e7b8f7aca55694b8a0
    def get_inputs(self):
        return [
            paddle.uniform([4205, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4205, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_aa323162a0a56599e198e83bd362f640(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_72553509889f68e7b8f7aca55694b8a0
    def get_inputs(self):
        return [
            paddle.uniform([4205, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4205, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_aa323162a0a56599e198e83bd362f640(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_72553509889f68e7b8f7aca55694b8a0
    def get_inputs(self):
        return [
            paddle.uniform([4205, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4205, 1], dtype='float32', min=0, max=0.5),
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