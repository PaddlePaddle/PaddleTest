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


class TestPrimitiveOp_1a248fc04866cf2bfae8ab7ccbb5ac67(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_791985e0863360b3b07d19d88c268b58
    def get_inputs(self):
        return [
            paddle.uniform([1745, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1745, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_1a248fc04866cf2bfae8ab7ccbb5ac67(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_791985e0863360b3b07d19d88c268b58
    def get_inputs(self):
        return [
            paddle.uniform([1745, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1745, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_1a248fc04866cf2bfae8ab7ccbb5ac67(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_791985e0863360b3b07d19d88c268b58
    def get_inputs(self):
        return [
            paddle.uniform([1745, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1745, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_1a248fc04866cf2bfae8ab7ccbb5ac67(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_791985e0863360b3b07d19d88c268b58
    def get_inputs(self):
        return [
            paddle.uniform([1745, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1745, 1], dtype='float32', min=0, max=0.5),
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


class TestPrimitiveOp_fbe6500dd59ecf807b45f0680970f438(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_791985e0863360b3b07d19d88c268b58
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.30527326464653015], [0.07021355628967285], [0.34596216678619385], [0.2309519201517105], [0.05144184082746506], [0.09776482731103897], [0.29527753591537476], [0.17036089301109314], [0.15745829045772552]], dtype='float32').reshape([9, 1]),
            paddle.to_tensor([[0.36898699402809143], [0.06626627594232559], [0.12720561027526855], [0.3147180676460266], [0.06824018061161041], [0.3909800946712494], [0.35480907559394836], [0.016808612272143364], [0.32742995023727417]], dtype='float32').reshape([9, 1]),
        ]


class TestPrimitiveOp_d8eb86f9ff1664f6a3d6da27c0af4bb2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_791985e0863360b3b07d19d88c268b58
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.06501298397779465], [0.03647551313042641], [0.3966253995895386], [0.3700411021709442], [0.20033740997314453], [0.13268493115901947], [0.20981046557426453], [0.3208114802837372], [0.22588470578193665]], dtype='float32').reshape([9, 1]),
            paddle.to_tensor([[0.1222463995218277], [0.36551961302757263], [0.1967533528804779], [0.3386157155036926], [0.08081060647964478], [0.049536582082509995], [0.14338451623916626], [0.04308648407459259], [0.22876402735710144]], dtype='float32').reshape([9, 1]),
        ]


class TestPrimitiveOp_d4e6bf75f8fd721777cdd5df64fbeee9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_791985e0863360b3b07d19d88c268b58
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.14627687633037567], [0.07964449375867844], [0.33320513367652893], [0.4071856439113617], [0.3619658946990967], [0.43402859568595886], [0.4941118359565735], [0.24435442686080933], [0.40847325325012207]], dtype='float32').reshape([9, 1]),
            paddle.to_tensor([[0.19455237686634064], [0.35046592354774475], [0.05605011805891991], [0.15758116543293], [0.41687849164009094], [0.35822534561157227], [0.07008229196071625], [0.4353342056274414], [0.1124526634812355]], dtype='float32').reshape([9, 1]),
        ]


class TestPrimitiveOp_7b356e0a269c15989f54c22c94544887(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_791985e0863360b3b07d19d88c268b58
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.2213127315044403], [0.13248451054096222], [0.12742774188518524], [0.18233563005924225], [0.41723570227622986], [0.06383621692657471], [0.40079569816589355], [0.42756208777427673], [0.2447492927312851]], dtype='float32').reshape([9, 1]),
            paddle.to_tensor([[0.446295827627182], [0.16909025609493256], [0.38718894124031067], [0.32923561334609985], [0.3280884623527527], [0.4127715229988098], [0.1773354858160019], [0.4922620356082916], [0.3027809262275696]], dtype='float32').reshape([9, 1]),
        ]


class TestPrimitiveOp_6bed96d79eaf398390fdc6d926930be3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_791985e0863360b3b07d19d88c268b58
    def get_inputs(self):
        return [
            paddle.uniform([5556, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([5556, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6bed96d79eaf398390fdc6d926930be3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_791985e0863360b3b07d19d88c268b58
    def get_inputs(self):
        return [
            paddle.uniform([5556, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([5556, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6bed96d79eaf398390fdc6d926930be3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_791985e0863360b3b07d19d88c268b58
    def get_inputs(self):
        return [
            paddle.uniform([5556, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([5556, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6bed96d79eaf398390fdc6d926930be3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_791985e0863360b3b07d19d88c268b58
    def get_inputs(self):
        return [
            paddle.uniform([5556, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([5556, 1], dtype='float32', min=0, max=0.5),
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


class TestPrimitiveOp_db45bd8ed72d3e77482550790c31c331(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9ac2539e1bfebdbbdaf888ff0d4461d3
    def get_inputs(self):
        return [
            paddle.to_tensor([0.06802684813737869, 0.3697265684604645, 0.02440202794969082, 0.18332761526107788, 0.2911740839481354, 0.12811702489852905], dtype='float32').reshape([6]),
            paddle.to_tensor([0.05024552345275879, 0.39770445227622986, 0.47278454899787903, 0.15071414411067963, 0.23610089719295502, 0.47662022709846497], dtype='float32').reshape([6]),
        ]


class TestPrimitiveOp_d3b436d8cab36a3e0c88192a98143901(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9ac2539e1bfebdbbdaf888ff0d4461d3
    def get_inputs(self):
        return [
            paddle.to_tensor([0.2703670859336853, 0.049813900142908096, 0.4583689868450165, 0.21031175553798676, 0.43299606442451477, 0.14105597138404846], dtype='float32').reshape([6]),
            paddle.to_tensor([0.29055315256118774, 0.4195096492767334, 0.1857345551252365, 0.33264780044555664, 0.3535158038139343, 0.09682358801364899], dtype='float32').reshape([6]),
        ]


class TestPrimitiveOp_0556c8c266b5420f82dfe0a27aaabcf3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9ac2539e1bfebdbbdaf888ff0d4461d3
    def get_inputs(self):
        return [
            paddle.to_tensor([0.06802684813737869, 0.3697265684604645, 0.02440202794969082, 0.18332761526107788, 0.2911740839481354, 0.12811702489852905], dtype='float32').reshape([6]),
            paddle.to_tensor([0.05370360612869263, 0.2583759129047394, 0.02003302238881588, 0.032925594598054886, 0.4194084107875824, 0.24436035752296448], dtype='float32').reshape([6]),
        ]


class TestPrimitiveOp_aefc8f8d7c8fab11cbf065b692969025(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9ac2539e1bfebdbbdaf888ff0d4461d3
    def get_inputs(self):
        return [
            paddle.to_tensor([0.2703670859336853, 0.049813900142908096, 0.4583689868450165, 0.21031175553798676, 0.43299606442451477, 0.14105597138404846], dtype='float32').reshape([6]),
            paddle.to_tensor([0.23499614000320435, 0.061109770089387894, 0.17594373226165771, 0.2823810279369354, 0.29694291949272156, 0.009236985817551613], dtype='float32').reshape([6]),
        ]


class TestPrimitiveOp_6bbe95818c37ca95b115711ba2d2802f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9ac2539e1bfebdbbdaf888ff0d4461d3
    def get_inputs(self):
        return [
            paddle.to_tensor([0.06802684813737869, 0.39770445227622986, 0.47278454899787903, 0.18332761526107788, 0.2911740839481354, 0.47662022709846497], dtype='float32').reshape([6]),
            paddle.to_tensor([0.10607437044382095, 0.45157837867736816, 0.14915001392364502, 0.35991302132606506, 0.15391811728477478, 0.4380766749382019], dtype='float32').reshape([6]),
        ]


class TestPrimitiveOp_0e8d2b457e4f6c19e22292487968fbc5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9ac2539e1bfebdbbdaf888ff0d4461d3
    def get_inputs(self):
        return [
            paddle.to_tensor([0.29055315256118774, 0.4195096492767334, 0.4583689868450165, 0.33264780044555664, 0.43299606442451477, 0.14105597138404846], dtype='float32').reshape([6]),
            paddle.to_tensor([0.1136271134018898, 0.025614285841584206, 0.3276135325431824, 0.17587409913539886, 0.43374499678611755, 0.010036587715148926], dtype='float32').reshape([6]),
        ]


class TestPrimitiveOp_267e5d80b550e060c881a262ef20e905(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_791985e0863360b3b07d19d88c268b58
    def get_inputs(self):
        return [
            paddle.uniform([1744, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1744, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_267e5d80b550e060c881a262ef20e905(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_791985e0863360b3b07d19d88c268b58
    def get_inputs(self):
        return [
            paddle.uniform([1744, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1744, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_267e5d80b550e060c881a262ef20e905(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_791985e0863360b3b07d19d88c268b58
    def get_inputs(self):
        return [
            paddle.uniform([1744, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1744, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_267e5d80b550e060c881a262ef20e905(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_791985e0863360b3b07d19d88c268b58
    def get_inputs(self):
        return [
            paddle.uniform([1744, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1744, 1], dtype='float32', min=0, max=0.5),
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


class TestPrimitiveOp_834b159461b58c167b147667d6751e50(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_791985e0863360b3b07d19d88c268b58
    def get_inputs(self):
        return [
            paddle.uniform([1547, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1547, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_834b159461b58c167b147667d6751e50(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_791985e0863360b3b07d19d88c268b58
    def get_inputs(self):
        return [
            paddle.uniform([1547, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1547, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_834b159461b58c167b147667d6751e50(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_791985e0863360b3b07d19d88c268b58
    def get_inputs(self):
        return [
            paddle.uniform([1547, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1547, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_834b159461b58c167b147667d6751e50(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_791985e0863360b3b07d19d88c268b58
    def get_inputs(self):
        return [
            paddle.uniform([1547, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1547, 1], dtype='float32', min=0, max=0.5),
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


class TestPrimitiveOp_a65a3cac8e5183b7667f6b80351b4dcf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_791985e0863360b3b07d19d88c268b58
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.061764106154441833]], dtype='float32').reshape([1, 1]),
            paddle.to_tensor([[0.2048177421092987]], dtype='float32').reshape([1, 1]),
        ]


class TestPrimitiveOp_c1b097814b54c65d7ec351c98c8c02eb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_791985e0863360b3b07d19d88c268b58
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.19873455166816711]], dtype='float32').reshape([1, 1]),
            paddle.to_tensor([[0.005689022596925497]], dtype='float32').reshape([1, 1]),
        ]


class TestPrimitiveOp_2b7357f6fda35ab3a855bfb610dfa546(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_791985e0863360b3b07d19d88c268b58
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.385765939950943]], dtype='float32').reshape([1, 1]),
            paddle.to_tensor([[0.0789174735546112]], dtype='float32').reshape([1, 1]),
        ]


class TestPrimitiveOp_23ec395cbc090b094262f72bf4aae24a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_791985e0863360b3b07d19d88c268b58
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.4691363573074341]], dtype='float32').reshape([1, 1]),
            paddle.to_tensor([[0.0012605844531208277]], dtype='float32').reshape([1, 1]),
        ]


class TestPrimitiveOp_4a505903f4d985b49698e5ca220837c1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_791985e0863360b3b07d19d88c268b58
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.3666752576828003], [0.19270555675029755], [0.09260793775320053], [0.18456070125102997], [0.4011060297489166], [0.3782949447631836]], dtype='float32').reshape([6, 1]),
            paddle.to_tensor([[0.0866883173584938], [0.13139678537845612], [0.3988319933414459], [0.4834783375263214], [0.05863175541162491], [0.11190173774957657]], dtype='float32').reshape([6, 1]),
        ]


class TestPrimitiveOp_4f2c862aed9bca28279d3d1500b493cc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_791985e0863360b3b07d19d88c268b58
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.37516072392463684], [0.3007451593875885], [0.010893420316278934], [0.08891236782073975], [0.005648438353091478], [0.16997677087783813]], dtype='float32').reshape([6, 1]),
            paddle.to_tensor([[0.44152939319610596], [0.4197368919849396], [0.18639807403087616], [0.18250931799411774], [0.45314154028892517], [0.34045061469078064]], dtype='float32').reshape([6, 1]),
        ]


class TestPrimitiveOp_c028d36978141ad19fc5666b1ea418aa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_791985e0863360b3b07d19d88c268b58
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.2444322109222412], [0.05394706130027771], [0.4057570695877075], [0.33757326006889343], [0.24670080840587616], [0.3902941942214966]], dtype='float32').reshape([6, 1]),
            paddle.to_tensor([[0.3076860010623932], [0.029868239536881447], [0.09312184154987335], [0.29760172963142395], [0.14579564332962036], [0.46690183877944946]], dtype='float32').reshape([6, 1]),
        ]


class TestPrimitiveOp_a1b7a9dafc2d339c94f55fa8cc7dac6e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_791985e0863360b3b07d19d88c268b58
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.49187493324279785], [0.36524298787117004], [0.07922337204217911], [0.4743187427520752], [0.0014592537190765142], [0.0375727079808712]], dtype='float32').reshape([6, 1]),
            paddle.to_tensor([[0.05468965321779251], [0.3557642102241516], [0.08949944376945496], [0.2927427291870117], [0.0025271896738559008], [0.2078334242105484]], dtype='float32').reshape([6, 1]),
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


class TestPrimitiveOp_1d5a800015f2d2da0e430540ef7041ab(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_791985e0863360b3b07d19d88c268b58
    def get_inputs(self):
        return [
            paddle.uniform([2056, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2056, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_1d5a800015f2d2da0e430540ef7041ab(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_791985e0863360b3b07d19d88c268b58
    def get_inputs(self):
        return [
            paddle.uniform([2056, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2056, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_1d5a800015f2d2da0e430540ef7041ab(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_791985e0863360b3b07d19d88c268b58
    def get_inputs(self):
        return [
            paddle.uniform([2056, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2056, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_1d5a800015f2d2da0e430540ef7041ab(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_791985e0863360b3b07d19d88c268b58
    def get_inputs(self):
        return [
            paddle.uniform([2056, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2056, 1], dtype='float32', min=0, max=0.5),
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


class TestPrimitiveOp_111c4d0965922f3e65aba56a0a06dbb8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_791985e0863360b3b07d19d88c268b58
    def get_inputs(self):
        return [
            paddle.uniform([4650, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4650, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_111c4d0965922f3e65aba56a0a06dbb8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_791985e0863360b3b07d19d88c268b58
    def get_inputs(self):
        return [
            paddle.uniform([4650, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4650, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_111c4d0965922f3e65aba56a0a06dbb8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_791985e0863360b3b07d19d88c268b58
    def get_inputs(self):
        return [
            paddle.uniform([4650, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4650, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_111c4d0965922f3e65aba56a0a06dbb8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_791985e0863360b3b07d19d88c268b58
    def get_inputs(self):
        return [
            paddle.uniform([4650, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4650, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4cd3f241d98d110d05d08b322079bb49(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_791985e0863360b3b07d19d88c268b58
    def get_inputs(self):
        return [
            paddle.uniform([1059, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1059, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4cd3f241d98d110d05d08b322079bb49(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_791985e0863360b3b07d19d88c268b58
    def get_inputs(self):
        return [
            paddle.uniform([1059, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1059, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4cd3f241d98d110d05d08b322079bb49(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_791985e0863360b3b07d19d88c268b58
    def get_inputs(self):
        return [
            paddle.uniform([1059, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1059, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4cd3f241d98d110d05d08b322079bb49(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_791985e0863360b3b07d19d88c268b58
    def get_inputs(self):
        return [
            paddle.uniform([1059, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1059, 1], dtype='float32', min=0, max=0.5),
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


class TestPrimitiveOp_bf1e8abeeaa7d2e2d7e3e706a617bae6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_791985e0863360b3b07d19d88c268b58
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.40543869137763977], [0.47366735339164734], [0.45203348994255066], [0.09204760938882828], [0.16088062524795532]], dtype='float32').reshape([5, 1]),
            paddle.to_tensor([[0.34268254041671753], [0.4942517578601837], [0.3020826578140259], [0.3473803997039795], [0.059936948120594025]], dtype='float32').reshape([5, 1]),
        ]


class TestPrimitiveOp_cc6f12af1afc1f30da317d1ab3e9935b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_791985e0863360b3b07d19d88c268b58
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.14999118447303772], [0.25430768728256226], [0.45005354285240173], [0.4590291678905487], [0.3022708296775818]], dtype='float32').reshape([5, 1]),
            paddle.to_tensor([[0.34039562940597534], [0.09946838766336441], [0.26270535588264465], [0.1753489077091217], [0.06309731304645538]], dtype='float32').reshape([5, 1]),
        ]


class TestPrimitiveOp_9d6e487fce06dc6e6cdd8835f536332c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_791985e0863360b3b07d19d88c268b58
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.034823983907699585], [0.024908585473895073], [0.3305168151855469], [0.3927406668663025], [0.1830960065126419]], dtype='float32').reshape([5, 1]),
            paddle.to_tensor([[0.35219553112983704], [0.37796422839164734], [0.45025119185447693], [0.43181949853897095], [0.0862424373626709]], dtype='float32').reshape([5, 1]),
        ]


class TestPrimitiveOp_8664ed0deb75369353a149da669d1cf9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_791985e0863360b3b07d19d88c268b58
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.4869818389415741], [0.2678717076778412], [0.3607769310474396], [0.23778222501277924], [0.26237985491752625]], dtype='float32').reshape([5, 1]),
            paddle.to_tensor([[0.4671558141708374], [0.34698235988616943], [0.27585870027542114], [0.10361013561487198], [0.2139054387807846]], dtype='float32').reshape([5, 1]),
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


class TestPrimitiveOp_73257efcc6610ac3cf8b37ad28a7bb8a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_791985e0863360b3b07d19d88c268b58
    def get_inputs(self):
        return [
            paddle.uniform([2347, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2347, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_73257efcc6610ac3cf8b37ad28a7bb8a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_791985e0863360b3b07d19d88c268b58
    def get_inputs(self):
        return [
            paddle.uniform([2347, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2347, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_73257efcc6610ac3cf8b37ad28a7bb8a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_791985e0863360b3b07d19d88c268b58
    def get_inputs(self):
        return [
            paddle.uniform([2347, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2347, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_73257efcc6610ac3cf8b37ad28a7bb8a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_791985e0863360b3b07d19d88c268b58
    def get_inputs(self):
        return [
            paddle.uniform([2347, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2347, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_341a7edea881a31cb47dad49d95379bd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_791985e0863360b3b07d19d88c268b58
    def get_inputs(self):
        return [
            paddle.uniform([3109, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3109, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_341a7edea881a31cb47dad49d95379bd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_791985e0863360b3b07d19d88c268b58
    def get_inputs(self):
        return [
            paddle.uniform([3109, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3109, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_341a7edea881a31cb47dad49d95379bd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_791985e0863360b3b07d19d88c268b58
    def get_inputs(self):
        return [
            paddle.uniform([3109, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3109, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_341a7edea881a31cb47dad49d95379bd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_791985e0863360b3b07d19d88c268b58
    def get_inputs(self):
        return [
            paddle.uniform([3109, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3109, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_01cee9713d8369c3a9a8e88c2bb5c7ae(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_791985e0863360b3b07d19d88c268b58
    def get_inputs(self):
        return [
            paddle.uniform([3813, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3813, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_01cee9713d8369c3a9a8e88c2bb5c7ae(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_791985e0863360b3b07d19d88c268b58
    def get_inputs(self):
        return [
            paddle.uniform([3813, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3813, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_01cee9713d8369c3a9a8e88c2bb5c7ae(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_791985e0863360b3b07d19d88c268b58
    def get_inputs(self):
        return [
            paddle.uniform([3813, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3813, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_01cee9713d8369c3a9a8e88c2bb5c7ae(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_791985e0863360b3b07d19d88c268b58
    def get_inputs(self):
        return [
            paddle.uniform([3813, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3813, 1], dtype='float32', min=0, max=0.5),
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


class TestPrimitiveOp_cfe2bdb5419e6faf5cea2a0ad1221d6a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_791985e0863360b3b07d19d88c268b58
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.15914317965507507], [0.45126867294311523], [0.333740234375], [0.38055622577667236]], dtype='float32').reshape([4, 1]),
            paddle.to_tensor([[0.057106561958789825], [0.3722459077835083], [0.06349729001522064], [0.3261486887931824]], dtype='float32').reshape([4, 1]),
        ]


class TestPrimitiveOp_206fb66ce4ea08df0b81e13fe78afc01(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_791985e0863360b3b07d19d88c268b58
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.4642404615879059], [0.014474418014287949], [0.04351218044757843], [0.07896348834037781]], dtype='float32').reshape([4, 1]),
            paddle.to_tensor([[0.3617795407772064], [0.36144939064979553], [0.43231305480003357], [0.17521435022354126]], dtype='float32').reshape([4, 1]),
        ]


class TestPrimitiveOp_8ed051a62283fcc417ee5a30ae98e864(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_791985e0863360b3b07d19d88c268b58
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.14417004585266113], [0.38891172409057617], [0.17883332073688507], [0.1194428876042366]], dtype='float32').reshape([4, 1]),
            paddle.to_tensor([[0.17582319676876068], [0.1984117180109024], [0.015906473621726036], [0.22741685807704926]], dtype='float32').reshape([4, 1]),
        ]


class TestPrimitiveOp_d0885f9dfa330a59b7c9fc27d553a9a5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_791985e0863360b3b07d19d88c268b58
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.2988428473472595], [0.4232253134250641], [0.06619598716497421], [0.38745006918907166]], dtype='float32').reshape([4, 1]),
            paddle.to_tensor([[0.2512052059173584], [0.3719286024570465], [0.23222172260284424], [0.10990522801876068]], dtype='float32').reshape([4, 1]),
        ]


class TestPrimitiveOp_6def8301e57c4d4b93c33f4ce8a6f3c5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_791985e0863360b3b07d19d88c268b58
    def get_inputs(self):
        return [
            paddle.uniform([2100, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2100, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6def8301e57c4d4b93c33f4ce8a6f3c5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_791985e0863360b3b07d19d88c268b58
    def get_inputs(self):
        return [
            paddle.uniform([2100, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2100, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6def8301e57c4d4b93c33f4ce8a6f3c5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_791985e0863360b3b07d19d88c268b58
    def get_inputs(self):
        return [
            paddle.uniform([2100, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2100, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6def8301e57c4d4b93c33f4ce8a6f3c5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_791985e0863360b3b07d19d88c268b58
    def get_inputs(self):
        return [
            paddle.uniform([2100, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2100, 1], dtype='float32', min=0, max=0.5),
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


class TestPrimitiveOp_b2ce02a2644ca76719154944e2c4db17(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_791985e0863360b3b07d19d88c268b58
    def get_inputs(self):
        return [
            paddle.uniform([4231, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4231, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b2ce02a2644ca76719154944e2c4db17(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_791985e0863360b3b07d19d88c268b58
    def get_inputs(self):
        return [
            paddle.uniform([4231, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4231, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b2ce02a2644ca76719154944e2c4db17(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_791985e0863360b3b07d19d88c268b58
    def get_inputs(self):
        return [
            paddle.uniform([4231, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4231, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b2ce02a2644ca76719154944e2c4db17(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_791985e0863360b3b07d19d88c268b58
    def get_inputs(self):
        return [
            paddle.uniform([4231, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4231, 1], dtype='float32', min=0, max=0.5),
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