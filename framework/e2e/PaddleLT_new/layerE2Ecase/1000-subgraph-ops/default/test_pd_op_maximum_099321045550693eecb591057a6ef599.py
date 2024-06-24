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


class TestPrimitiveOp_84b1b2975b4e2e30144599a74ab84370(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_72553509889f68e7b8f7aca55694b8a0
    def get_inputs(self):
        return [
            paddle.uniform([1841, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1841, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_84b1b2975b4e2e30144599a74ab84370(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_72553509889f68e7b8f7aca55694b8a0
    def get_inputs(self):
        return [
            paddle.uniform([1841, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1841, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_84b1b2975b4e2e30144599a74ab84370(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_72553509889f68e7b8f7aca55694b8a0
    def get_inputs(self):
        return [
            paddle.uniform([1841, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1841, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_84b1b2975b4e2e30144599a74ab84370(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_72553509889f68e7b8f7aca55694b8a0
    def get_inputs(self):
        return [
            paddle.uniform([1841, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1841, 1], dtype='float32', min=0, max=0.5),
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


class TestPrimitiveOp_12be6d31974d5f4fd38b87318cda966a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_791985e0863360b3b07d19d88c268b58
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.2136506289243698], [0.44700920581817627], [0.4920419454574585], [0.324871301651001], [0.4347423315048218], [0.11308283358812332], [0.4580673575401306], [0.2829873263835907], [0.09454172104597092]], dtype='float32').reshape([9, 1]),
            paddle.to_tensor([[0.05196747928857803], [0.3380136489868164], [0.4891405999660492], [0.2964899241924286], [0.20674164593219757], [0.010906912386417389], [0.49398526549339294], [0.1478980928659439], [0.371082603931427]], dtype='float32').reshape([9, 1]),
        ]


class TestPrimitiveOp_98459c6cb4c11cee6ad20f5acf2fcd56(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_791985e0863360b3b07d19d88c268b58
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.3358563482761383], [0.32271942496299744], [0.44343236088752747], [7.760437438264489e-05], [0.2671041786670685], [0.39811384677886963], [0.4129534661769867], [0.08367976546287537], [0.2420433908700943]], dtype='float32').reshape([9, 1]),
            paddle.to_tensor([[0.4841810464859009], [0.08505020290613174], [0.18903541564941406], [0.38031214475631714], [0.10913760215044022], [0.46841099858283997], [0.29409098625183105], [0.2900417149066925], [0.05474267899990082]], dtype='float32').reshape([9, 1]),
        ]


class TestPrimitiveOp_879612bb7cb60da149c1c5bf89715083(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_791985e0863360b3b07d19d88c268b58
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.48502784967422485], [0.007254623807966709], [0.08551525324583054], [0.28749170899391174], [0.42672285437583923], [0.4068581461906433], [0.00696443160995841], [0.11411301791667938], [0.016163386404514313]], dtype='float32').reshape([9, 1]),
            paddle.to_tensor([[0.22089730203151703], [0.033434219658374786], [0.16276678442955017], [0.3074219524860382], [0.03921782970428467], [0.38058915734291077], [0.19385917484760284], [0.09035753458738327], [0.05179273337125778]], dtype='float32').reshape([9, 1]),
        ]


class TestPrimitiveOp_1df530bae4745348ed548081b531f7cc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_791985e0863360b3b07d19d88c268b58
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.43891218304634094], [0.3814011812210083], [0.4228847026824951], [0.03740933537483215], [0.2268114984035492], [0.26206380128860474], [0.4565003216266632], [0.3727211654186249], [0.10162022709846497]], dtype='float32').reshape([9, 1]),
            paddle.to_tensor([[0.032230887562036514], [0.18736937642097473], [0.015003582462668419], [0.33160749077796936], [0.09220337867736816], [0.21914060413837433], [0.005615626461803913], [0.10680714249610901], [0.24876168370246887]], dtype='float32').reshape([9, 1]),
        ]


class TestPrimitiveOp_57e48c92c9ee0e59c5a3d175fa39d750(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_72553509889f68e7b8f7aca55694b8a0
    def get_inputs(self):
        return [
            paddle.uniform([5562, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([5562, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_57e48c92c9ee0e59c5a3d175fa39d750(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_72553509889f68e7b8f7aca55694b8a0
    def get_inputs(self):
        return [
            paddle.uniform([5562, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([5562, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_57e48c92c9ee0e59c5a3d175fa39d750(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_72553509889f68e7b8f7aca55694b8a0
    def get_inputs(self):
        return [
            paddle.uniform([5562, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([5562, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_57e48c92c9ee0e59c5a3d175fa39d750(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_72553509889f68e7b8f7aca55694b8a0
    def get_inputs(self):
        return [
            paddle.uniform([5562, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([5562, 1], dtype='float32', min=0, max=0.5),
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


class TestPrimitiveOp_a6173abf440a85b842381fd59bcc1b4b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9ac2539e1bfebdbbdaf888ff0d4461d3
    def get_inputs(self):
        return [
            paddle.to_tensor([0.14741972088813782, 0.1336534470319748, 0.11528350412845612, 0.4567588269710541, 0.24708279967308044, 0.3292865753173828], dtype='float32').reshape([6]),
            paddle.to_tensor([0.0693720355629921, 0.37168705463409424, 0.3194563388824463, 0.31869107484817505, 0.14553475379943848, 0.33666518330574036], dtype='float32').reshape([6]),
        ]


class TestPrimitiveOp_2dc085259b291799d643393469f8159d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9ac2539e1bfebdbbdaf888ff0d4461d3
    def get_inputs(self):
        return [
            paddle.to_tensor([0.25797852873802185, 0.12108806520700455, 0.2023414820432663, 0.36508259177207947, 0.0434037484228611, 0.07065112143754959], dtype='float32').reshape([6]),
            paddle.to_tensor([0.3068593740463257, 0.08607549220323563, 0.11235673725605011, 0.14383947849273682, 0.45763009786605835, 0.3360748291015625], dtype='float32').reshape([6]),
        ]


class TestPrimitiveOp_57707bfb3b1b0ca20eee58c7415f9c22(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9ac2539e1bfebdbbdaf888ff0d4461d3
    def get_inputs(self):
        return [
            paddle.to_tensor([0.14741972088813782, 0.1336534470319748, 0.11528350412845612, 0.4567588269710541, 0.24708279967308044, 0.3292865753173828], dtype='float32').reshape([6]),
            paddle.to_tensor([0.2984810769557953, 0.018444553017616272, 0.06930149346590042, 0.3149198293685913, 0.005490172654390335, 0.3335161805152893], dtype='float32').reshape([6]),
        ]


class TestPrimitiveOp_d57e783ca26e75edd12cf31a8abcf50e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9ac2539e1bfebdbbdaf888ff0d4461d3
    def get_inputs(self):
        return [
            paddle.to_tensor([0.25797852873802185, 0.12108806520700455, 0.2023414820432663, 0.36508259177207947, 0.0434037484228611, 0.07065112143754959], dtype='float32').reshape([6]),
            paddle.to_tensor([0.2929052710533142, 0.40369516611099243, 0.1233789473772049, 0.3182181119918823, 0.267822802066803, 0.4557313323020935], dtype='float32').reshape([6]),
        ]


class TestPrimitiveOp_b1a4966e5f637549d3aa0686ec3f2a55(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9ac2539e1bfebdbbdaf888ff0d4461d3
    def get_inputs(self):
        return [
            paddle.to_tensor([0.14741972088813782, 0.37168705463409424, 0.3194563388824463, 0.4567588269710541, 0.24708279967308044, 0.33666518330574036], dtype='float32').reshape([6]),
            paddle.to_tensor([0.22594380378723145, 0.10565175116062164, 0.03872787579894066, 0.2721693217754364, 0.14351142942905426, 0.3796778917312622], dtype='float32').reshape([6]),
        ]


class TestPrimitiveOp_6aff3c94790c95678f3938d5e9ff2912(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9ac2539e1bfebdbbdaf888ff0d4461d3
    def get_inputs(self):
        return [
            paddle.to_tensor([0.3068593740463257, 0.12108806520700455, 0.2023414820432663, 0.36508259177207947, 0.45763009786605835, 0.3360748291015625], dtype='float32').reshape([6]),
            paddle.to_tensor([0.1421709507703781, 0.1708022505044937, 0.1709064394235611, 0.046536415815353394, 0.06752748787403107, 0.041682589799165726], dtype='float32').reshape([6]),
        ]


class TestPrimitiveOp_c1a0828abf6d5426f8f557bac3350a7b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_72553509889f68e7b8f7aca55694b8a0
    def get_inputs(self):
        return [
            paddle.uniform([1734, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1734, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c1a0828abf6d5426f8f557bac3350a7b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_72553509889f68e7b8f7aca55694b8a0
    def get_inputs(self):
        return [
            paddle.uniform([1734, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1734, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c1a0828abf6d5426f8f557bac3350a7b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_72553509889f68e7b8f7aca55694b8a0
    def get_inputs(self):
        return [
            paddle.uniform([1734, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1734, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c1a0828abf6d5426f8f557bac3350a7b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_72553509889f68e7b8f7aca55694b8a0
    def get_inputs(self):
        return [
            paddle.uniform([1734, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1734, 1], dtype='float32', min=0, max=0.5),
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


class TestPrimitiveOp_29c84d102a002131823337eb3b185d4b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_72553509889f68e7b8f7aca55694b8a0
    def get_inputs(self):
        return [
            paddle.uniform([1541, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1541, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_29c84d102a002131823337eb3b185d4b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_72553509889f68e7b8f7aca55694b8a0
    def get_inputs(self):
        return [
            paddle.uniform([1541, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1541, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_29c84d102a002131823337eb3b185d4b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_72553509889f68e7b8f7aca55694b8a0
    def get_inputs(self):
        return [
            paddle.uniform([1541, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1541, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_29c84d102a002131823337eb3b185d4b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_72553509889f68e7b8f7aca55694b8a0
    def get_inputs(self):
        return [
            paddle.uniform([1541, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1541, 1], dtype='float32', min=0, max=0.5),
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


class TestPrimitiveOp_5e50d8ff36f6256157d9f01186edb123(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_791985e0863360b3b07d19d88c268b58
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.26397430896759033]], dtype='float32').reshape([1, 1]),
            paddle.to_tensor([[0.46222788095474243]], dtype='float32').reshape([1, 1]),
        ]


class TestPrimitiveOp_601bb5ffa176e3ac496ca2d77ac2511d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_791985e0863360b3b07d19d88c268b58
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.25180718302726746]], dtype='float32').reshape([1, 1]),
            paddle.to_tensor([[0.13167576491832733]], dtype='float32').reshape([1, 1]),
        ]


class TestPrimitiveOp_87f938e5e7ffa07af2c97726aef2ed15(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_791985e0863360b3b07d19d88c268b58
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.30370238423347473]], dtype='float32').reshape([1, 1]),
            paddle.to_tensor([[0.17654746770858765]], dtype='float32').reshape([1, 1]),
        ]


class TestPrimitiveOp_f558ba9f40b05ef5c727ce70637edde1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_791985e0863360b3b07d19d88c268b58
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.34866175055503845]], dtype='float32').reshape([1, 1]),
            paddle.to_tensor([[0.11643274873495102]], dtype='float32').reshape([1, 1]),
        ]


class TestPrimitiveOp_5cc8649397bac27022f9d4e53040f93f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_791985e0863360b3b07d19d88c268b58
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.3404685854911804], [0.3644312620162964], [0.44551825523376465], [0.496250182390213], [0.2719564139842987], [0.07249618321657181]], dtype='float32').reshape([6, 1]),
            paddle.to_tensor([[0.03989572077989578], [0.06823953241109848], [0.13498272001743317], [0.24980278313159943], [0.27458834648132324], [0.18564586341381073]], dtype='float32').reshape([6, 1]),
        ]


class TestPrimitiveOp_a0913d0c81a4a3f39f9f035550bf8db7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_791985e0863360b3b07d19d88c268b58
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.3674864172935486], [0.35537904500961304], [0.435450941324234], [0.12282329052686691], [0.16958539187908173], [0.04632904753088951]], dtype='float32').reshape([6, 1]),
            paddle.to_tensor([[0.44728875160217285], [0.1407179981470108], [0.16133081912994385], [0.4250192940235138], [0.3090977668762207], [0.028634952381253242]], dtype='float32').reshape([6, 1]),
        ]


class TestPrimitiveOp_e5e71a5b1366727c038be68e43250580(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_791985e0863360b3b07d19d88c268b58
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.332610547542572], [0.14597834646701813], [0.19613602757453918], [0.12908796966075897], [0.26270008087158203], [0.133039191365242]], dtype='float32').reshape([6, 1]),
            paddle.to_tensor([[0.3089545667171478], [0.24098819494247437], [0.3093312382698059], [0.2416381686925888], [0.2358388751745224], [0.31183764338493347]], dtype='float32').reshape([6, 1]),
        ]


class TestPrimitiveOp_fe92a452cac60424e7f1098e805adc8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_791985e0863360b3b07d19d88c268b58
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.40432968735694885], [0.2534337639808655], [0.18960238993167877], [0.41409391164779663], [0.25769028067588806], [0.23563654720783234]], dtype='float32').reshape([6, 1]),
            paddle.to_tensor([[0.07759702950716019], [0.23825453221797943], [0.15273594856262207], [0.2640349566936493], [0.366540789604187], [0.189493328332901]], dtype='float32').reshape([6, 1]),
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


class TestPrimitiveOp_4b5286fa24d5e44f0f4f6a0cde0b9da0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_72553509889f68e7b8f7aca55694b8a0
    def get_inputs(self):
        return [
            paddle.uniform([2061, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2061, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4b5286fa24d5e44f0f4f6a0cde0b9da0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_72553509889f68e7b8f7aca55694b8a0
    def get_inputs(self):
        return [
            paddle.uniform([2061, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2061, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4b5286fa24d5e44f0f4f6a0cde0b9da0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_72553509889f68e7b8f7aca55694b8a0
    def get_inputs(self):
        return [
            paddle.uniform([2061, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2061, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4b5286fa24d5e44f0f4f6a0cde0b9da0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_72553509889f68e7b8f7aca55694b8a0
    def get_inputs(self):
        return [
            paddle.uniform([2061, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2061, 1], dtype='float32', min=0, max=0.5),
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


class TestPrimitiveOp_ff37e54eee7c80b2fe2e58561bc591b1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_72553509889f68e7b8f7aca55694b8a0
    def get_inputs(self):
        return [
            paddle.uniform([4642, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4642, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ff37e54eee7c80b2fe2e58561bc591b1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_72553509889f68e7b8f7aca55694b8a0
    def get_inputs(self):
        return [
            paddle.uniform([4642, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4642, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ff37e54eee7c80b2fe2e58561bc591b1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_72553509889f68e7b8f7aca55694b8a0
    def get_inputs(self):
        return [
            paddle.uniform([4642, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4642, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ff37e54eee7c80b2fe2e58561bc591b1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_72553509889f68e7b8f7aca55694b8a0
    def get_inputs(self):
        return [
            paddle.uniform([4642, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4642, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_14fb654e4bff1253be4374925d373521(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_72553509889f68e7b8f7aca55694b8a0
    def get_inputs(self):
        return [
            paddle.uniform([1042, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1042, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_14fb654e4bff1253be4374925d373521(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_72553509889f68e7b8f7aca55694b8a0
    def get_inputs(self):
        return [
            paddle.uniform([1042, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1042, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_14fb654e4bff1253be4374925d373521(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_72553509889f68e7b8f7aca55694b8a0
    def get_inputs(self):
        return [
            paddle.uniform([1042, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1042, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_14fb654e4bff1253be4374925d373521(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_72553509889f68e7b8f7aca55694b8a0
    def get_inputs(self):
        return [
            paddle.uniform([1042, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1042, 1], dtype='float32', min=0, max=0.5),
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


class TestPrimitiveOp_7dfa0d58a2f7db045a02ffe319e1e0a4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_791985e0863360b3b07d19d88c268b58
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.1571986973285675], [0.11458301544189453], [0.34768521785736084], [0.20442333817481995], [0.04476496949791908]], dtype='float32').reshape([5, 1]),
            paddle.to_tensor([[0.008164001628756523], [0.1091982051730156], [0.34744080901145935], [0.31836310029029846], [0.16771680116653442]], dtype='float32').reshape([5, 1]),
        ]


class TestPrimitiveOp_62dbfa488ac22c8e7c9cf186f176ee63(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_791985e0863360b3b07d19d88c268b58
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.3675876557826996], [0.2628155052661896], [0.3304205536842346], [0.4551815986633301], [0.469809353351593]], dtype='float32').reshape([5, 1]),
            paddle.to_tensor([[0.2649228572845459], [0.3236313760280609], [0.3462056517601013], [0.14429233968257904], [0.2508985102176666]], dtype='float32').reshape([5, 1]),
        ]


class TestPrimitiveOp_2920912921cee1f75a1de07606050d3e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_791985e0863360b3b07d19d88c268b58
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.4107484221458435], [0.2350795716047287], [0.32436296343803406], [0.33859461545944214], [0.17683643102645874]], dtype='float32').reshape([5, 1]),
            paddle.to_tensor([[0.00047272868687286973], [0.056612178683280945], [0.17818866670131683], [0.2440360188484192], [0.006049283314496279]], dtype='float32').reshape([5, 1]),
        ]


class TestPrimitiveOp_b9fcb692d20b9adde0c51f1c176756cd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_791985e0863360b3b07d19d88c268b58
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.2710123062133789], [0.24374403059482574], [0.10803781449794769], [0.17378324270248413], [0.29924002289772034]], dtype='float32').reshape([5, 1]),
            paddle.to_tensor([[0.049238573759794235], [0.14470472931861877], [0.00861063040792942], [0.40792182087898254], [0.3668667674064636]], dtype='float32').reshape([5, 1]),
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


class TestPrimitiveOp_7a96e87183dce60ac933a4211b14db51(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_72553509889f68e7b8f7aca55694b8a0
    def get_inputs(self):
        return [
            paddle.uniform([2369, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2369, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_7a96e87183dce60ac933a4211b14db51(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_72553509889f68e7b8f7aca55694b8a0
    def get_inputs(self):
        return [
            paddle.uniform([2369, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2369, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_7a96e87183dce60ac933a4211b14db51(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_72553509889f68e7b8f7aca55694b8a0
    def get_inputs(self):
        return [
            paddle.uniform([2369, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2369, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_7a96e87183dce60ac933a4211b14db51(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_72553509889f68e7b8f7aca55694b8a0
    def get_inputs(self):
        return [
            paddle.uniform([2369, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2369, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_fd35af21b95fba7932b7934781744f5f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_72553509889f68e7b8f7aca55694b8a0
    def get_inputs(self):
        return [
            paddle.uniform([3054, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3054, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_fd35af21b95fba7932b7934781744f5f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_72553509889f68e7b8f7aca55694b8a0
    def get_inputs(self):
        return [
            paddle.uniform([3054, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3054, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_fd35af21b95fba7932b7934781744f5f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_72553509889f68e7b8f7aca55694b8a0
    def get_inputs(self):
        return [
            paddle.uniform([3054, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3054, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_fd35af21b95fba7932b7934781744f5f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_72553509889f68e7b8f7aca55694b8a0
    def get_inputs(self):
        return [
            paddle.uniform([3054, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3054, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d2ec2363eb8c8c7ff53dcae947d9d906(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_72553509889f68e7b8f7aca55694b8a0
    def get_inputs(self):
        return [
            paddle.uniform([3819, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3819, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d2ec2363eb8c8c7ff53dcae947d9d906(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_72553509889f68e7b8f7aca55694b8a0
    def get_inputs(self):
        return [
            paddle.uniform([3819, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3819, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d2ec2363eb8c8c7ff53dcae947d9d906(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_72553509889f68e7b8f7aca55694b8a0
    def get_inputs(self):
        return [
            paddle.uniform([3819, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3819, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d2ec2363eb8c8c7ff53dcae947d9d906(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_72553509889f68e7b8f7aca55694b8a0
    def get_inputs(self):
        return [
            paddle.uniform([3819, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3819, 1], dtype='float32', min=0, max=0.5),
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


class TestPrimitiveOp_dbf4a4f84521a32b3f62ffb91f0febed(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_791985e0863360b3b07d19d88c268b58
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.06872285902500153], [0.06164400279521942], [0.20687055587768555], [0.3251105844974518]], dtype='float32').reshape([4, 1]),
            paddle.to_tensor([[0.4831085503101349], [0.4750896692276001], [0.1859712302684784], [0.05235862359404564]], dtype='float32').reshape([4, 1]),
        ]


class TestPrimitiveOp_9da8673fc0211053bacf156e6e292b2f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_791985e0863360b3b07d19d88c268b58
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.32444852590560913], [0.14037735760211945], [0.08293063938617706], [0.08842404931783676]], dtype='float32').reshape([4, 1]),
            paddle.to_tensor([[0.31226545572280884], [0.035297587513923645], [0.2893640995025635], [0.003905576653778553]], dtype='float32').reshape([4, 1]),
        ]


class TestPrimitiveOp_757776e958be5f0461897c9800812cd3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_791985e0863360b3b07d19d88c268b58
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.3838992714881897], [0.1253899484872818], [0.24463814496994019], [0.23376497626304626]], dtype='float32').reshape([4, 1]),
            paddle.to_tensor([[0.27045926451683044], [0.49347126483917236], [0.4834122955799103], [0.4336952865123749]], dtype='float32').reshape([4, 1]),
        ]


class TestPrimitiveOp_c0c2d3c01432d06ed0c8094df6988293(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_791985e0863360b3b07d19d88c268b58
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.19007621705532074], [0.4756835699081421], [0.2541756331920624], [0.4143301248550415]], dtype='float32').reshape([4, 1]),
            paddle.to_tensor([[0.04296104982495308], [0.3528343439102173], [0.31736236810684204], [0.22007668018341064]], dtype='float32').reshape([4, 1]),
        ]


class TestPrimitiveOp_70f0d751f58df752f1d0fa22cb6aaa30(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_72553509889f68e7b8f7aca55694b8a0
    def get_inputs(self):
        return [
            paddle.uniform([2092, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2092, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_70f0d751f58df752f1d0fa22cb6aaa30(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_72553509889f68e7b8f7aca55694b8a0
    def get_inputs(self):
        return [
            paddle.uniform([2092, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2092, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_70f0d751f58df752f1d0fa22cb6aaa30(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_72553509889f68e7b8f7aca55694b8a0
    def get_inputs(self):
        return [
            paddle.uniform([2092, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2092, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_70f0d751f58df752f1d0fa22cb6aaa30(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_72553509889f68e7b8f7aca55694b8a0
    def get_inputs(self):
        return [
            paddle.uniform([2092, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2092, 1], dtype='float32', min=0, max=0.5),
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


class TestPrimitiveOp_e0c267511423bcb6d1a1cb39735bea84(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_72553509889f68e7b8f7aca55694b8a0
    def get_inputs(self):
        return [
            paddle.uniform([4214, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4214, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e0c267511423bcb6d1a1cb39735bea84(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_72553509889f68e7b8f7aca55694b8a0
    def get_inputs(self):
        return [
            paddle.uniform([4214, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4214, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e0c267511423bcb6d1a1cb39735bea84(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_72553509889f68e7b8f7aca55694b8a0
    def get_inputs(self):
        return [
            paddle.uniform([4214, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4214, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e0c267511423bcb6d1a1cb39735bea84(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_72553509889f68e7b8f7aca55694b8a0
    def get_inputs(self):
        return [
            paddle.uniform([4214, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4214, 1], dtype='float32', min=0, max=0.5),
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