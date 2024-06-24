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



class PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_c806fc34a2e4e2958d32485bf1c44917(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.uniform([1, 144, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_1c21e9be9a7e71dc79ebb27483a54b38(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_8615b43e596283b8e3d73902c3fc3076(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1c21e9be9a7e71dc79ebb27483a54b38
    def get_inputs(self):
        return [
            paddle.to_tensor([[5.034132957458496, 5.051319122314453, 5.2611212730407715, 4.580645561218262, 4.762973308563232, 5.002790451049805, 5.6654815673828125, 4.349248886108398, 5.136642932891846, 5.052401065826416, 4.996928691864014, 4.306968688964844, 4.845163345336914, 4.200080394744873, 5.610182285308838, 4.703643798828125, 4.884689807891846, 4.9336838722229]], dtype='float32').reshape([1, 18]),
        ]


class TestPrimitiveOp_0fca238c213111a947247f4e5a328883(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1c21e9be9a7e71dc79ebb27483a54b38
    def get_inputs(self):
        return [
            paddle.to_tensor([[7.229928970336914, 6.533698081970215, 6.732686519622803, 6.243114948272705, 7.0304975509643555, 6.9293904304504395, 6.767514228820801, 6.359036445617676, 5.74185037612915, 6.8545637130737305, 6.183698654174805, 6.503045558929443, 5.956207752227783, 6.541355133056641, 5.515870571136475, 5.586697101593018, 6.758831977844238, 7.21973991394043, 6.840029716491699, 6.680665016174316, 6.361820697784424, 6.038775444030762, 6.562516689300537]], dtype='float32').reshape([1, 23]),
        ]


class TestPrimitiveOp_823c928d12a943ccf0179ecff50ea704(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.uniform([1, 40, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a69bac1c7b78177869edecf93f82fa10(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1c21e9be9a7e71dc79ebb27483a54b38
    def get_inputs(self):
        return [
            paddle.uniform([1, 240], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d57a1aae47ed3c77764096b97c458c24(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1c21e9be9a7e71dc79ebb27483a54b38
    def get_inputs(self):
        return [
            paddle.uniform([1, 120], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_66b2e381f408968c540b43824c5ae77b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 20, 32], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d9b17ce2fbe91bd0358053a4b6f653a4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1c21e9be9a7e71dc79ebb27483a54b38
    def get_inputs(self):
        return [
            paddle.uniform([512, 1024], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d9b17ce2fbe91bd0358053a4b6f653a4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1c21e9be9a7e71dc79ebb27483a54b38
    def get_inputs(self):
        return [
            paddle.uniform([512, 1024], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_fb8f67f9e9c1678f4f4c8b0e74def09f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.uniform([1, 168, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b2d8d69fad67c4180269ac3bf4ce6335(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[7.878262519836426]], [[7.868306636810303]], [[7.780619144439697]], [[7.209699630737305]], [[7.433918476104736]], [[7.6283040046691895]], [[7.112617492675781]], [[7.02634334564209]], [[7.322437286376953]], [[8.220512390136719]], [[7.710140228271484]], [[7.590466499328613]], [[7.6705546379089355]], [[8.223761558532715]], [[7.7061309814453125]], [[7.289975643157959]], [[7.402900218963623]], [[8.340198516845703]], [[7.1758904457092285]], [[7.918051242828369]], [[7.745652675628662]], [[8.17780876159668]], [[9.052896499633789]], [[7.247939586639404]], [[7.934576034545898]], [[7.615261077880859]], [[7.53016471862793]], [[7.217159748077393]], [[8.381265640258789]], [[7.74073600769043]]]], dtype='float32').reshape([1, 30, 1, 1]),
        ]


class TestPrimitiveOp_cec624690c57e95be692914c3a76b07a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1c21e9be9a7e71dc79ebb27483a54b38
    def get_inputs(self):
        return [
            paddle.uniform([10, 84], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3e4302736473aef40d5fa393830942b1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 80, 120], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3e4302736473aef40d5fa393830942b1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 80, 120], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3e4302736473aef40d5fa393830942b1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 80, 120], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3e4302736473aef40d5fa393830942b1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 80, 120], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3e4302736473aef40d5fa393830942b1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 80, 120], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3e4302736473aef40d5fa393830942b1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 80, 120], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3e4302736473aef40d5fa393830942b1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 80, 120], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3e4302736473aef40d5fa393830942b1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 80, 120], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e705312a45569a627ea6f44159272738(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 40, 60], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e705312a45569a627ea6f44159272738(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 40, 60], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e705312a45569a627ea6f44159272738(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 40, 60], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e705312a45569a627ea6f44159272738(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 40, 60], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e705312a45569a627ea6f44159272738(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 40, 60], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e705312a45569a627ea6f44159272738(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 40, 60], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e705312a45569a627ea6f44159272738(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 40, 60], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e705312a45569a627ea6f44159272738(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 40, 60], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b5ff066e0e95f1fbcd21db3b8d335f9c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 20, 30], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b5ff066e0e95f1fbcd21db3b8d335f9c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 20, 30], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b5ff066e0e95f1fbcd21db3b8d335f9c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 20, 30], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b5ff066e0e95f1fbcd21db3b8d335f9c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 20, 30], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b5ff066e0e95f1fbcd21db3b8d335f9c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 20, 30], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b5ff066e0e95f1fbcd21db3b8d335f9c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 20, 30], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b5ff066e0e95f1fbcd21db3b8d335f9c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 20, 30], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b5ff066e0e95f1fbcd21db3b8d335f9c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 20, 30], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b3036a4887b77ccad8a52f82ca01261e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 10, 15], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b3036a4887b77ccad8a52f82ca01261e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 10, 15], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b3036a4887b77ccad8a52f82ca01261e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 10, 15], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b3036a4887b77ccad8a52f82ca01261e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 10, 15], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b3036a4887b77ccad8a52f82ca01261e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 10, 15], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b3036a4887b77ccad8a52f82ca01261e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 10, 15], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b3036a4887b77ccad8a52f82ca01261e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 10, 15], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b3036a4887b77ccad8a52f82ca01261e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 10, 15], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_2eab2b1824f596a4f5e1a35e10858673(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 5, 8], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_2eab2b1824f596a4f5e1a35e10858673(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 5, 8], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_2eab2b1824f596a4f5e1a35e10858673(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 5, 8], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_2eab2b1824f596a4f5e1a35e10858673(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 5, 8], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_2eab2b1824f596a4f5e1a35e10858673(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 5, 8], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_2eab2b1824f596a4f5e1a35e10858673(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 5, 8], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_2eab2b1824f596a4f5e1a35e10858673(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 5, 8], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_2eab2b1824f596a4f5e1a35e10858673(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 5, 8], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_597357d308167046b55da673c65dbe69(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[8.262385368347168]], [[7.8208513259887695]], [[8.153039932250977]], [[7.087778568267822]], [[7.932410717010498]], [[8.081872940063477]], [[7.86402702331543]], [[7.80531644821167]], [[7.08975887298584]], [[8.170979499816895]], [[7.904594898223877]], [[9.098105430603027]], [[8.166693687438965]], [[7.506779193878174]], [[8.451950073242188]], [[8.012106895446777]], [[8.47104263305664]], [[6.885653972625732]], [[6.992383003234863]], [[7.868102073669434]], [[8.112833976745605]], [[8.527132987976074]], [[7.738118648529053]], [[7.577332019805908]], [[8.22385025024414]], [[8.072021484375]], [[8.153665542602539]], [[6.797152996063232]], [[8.04731273651123]], [[8.220541954040527]]]], dtype='float32').reshape([1, 30, 1, 1]),
        ]


class TestPrimitiveOp_cb6e562ed929755429e92268d85b9633(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 50, 76], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4ad3681785ac53a64bc49563de00e4bc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[1.3657139539718628]], [[1.2159758806228638]], [[1.2202329635620117]], [[1.270909309387207]], [[1.1494554281234741]]]], dtype='float32').reshape([1, 5, 1, 1]),
        ]


class TestPrimitiveOp_47deb6226168ad3816fde72f4f6f25c4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[2.6581876277923584]], [[2.988999843597412]], [[2.862517833709717]], [[2.8966684341430664]], [[2.84088134765625]], [[2.676283121109009]], [[2.9992473125457764]], [[2.9194250106811523]], [[2.943911075592041]], [[2.860955238342285]]]], dtype='float32').reshape([1, 10, 1, 1]),
        ]


class TestPrimitiveOp_d43e57aa76dae90b491308a51147aeea(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.uniform([1, 240, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_21d045352b520e16d53194652d452352(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[6.257889747619629]], [[6.18993616104126]], [[6.881523132324219]], [[5.47817850112915]], [[5.700344085693359]], [[6.758537769317627]], [[6.853168487548828]], [[6.237391948699951]], [[6.216843605041504]], [[6.320457458496094]], [[6.477090835571289]], [[6.859305381774902]], [[6.186976909637451]], [[6.206911563873291]], [[6.260547161102295]], [[5.5532612800598145]], [[5.304288387298584]], [[6.289677619934082]], [[6.5896077156066895]], [[6.015058517456055]], [[7.109218597412109]], [[6.313172340393066]], [[6.01460075378418]], [[5.406548976898193]]]], dtype='float32').reshape([1, 24, 1, 1]),
        ]


class TestPrimitiveOp_46630bfb755a833585f9698cf9be7827(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 100, 152], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_38f2fd615db526e352bb46393710f73a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 13, 19], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_426e32b3f64f9f9fcea64832d60e1595(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1c21e9be9a7e71dc79ebb27483a54b38
    def get_inputs(self):
        return [
            paddle.uniform([10, 15], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6ea7d17b2295085a465d6a87af7e2ef3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[4.22940731048584]], [[5.363189220428467]], [[5.409626483917236]], [[4.828910827636719]], [[5.264454364776611]], [[4.615478515625]], [[4.274387359619141]], [[5.118419647216797]], [[4.849647045135498]], [[5.2210564613342285]], [[4.860780715942383]], [[5.749593734741211]], [[5.439781665802002]], [[4.932239055633545]], [[5.167477130889893]], [[5.709780216217041]], [[5.35679817199707]], [[4.660642623901367]]]], dtype='float32').reshape([1, 18, 1, 1]),
        ]


class TestPrimitiveOp_d43e57aa76dae90b491308a51147aeea(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.uniform([1, 240, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ac938442d7b2df1f44245392c4ae4462(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[6.02589750289917]], [[6.214494228363037]], [[6.199380874633789]], [[6.936142921447754]], [[5.8849263191223145]], [[7.262293815612793]], [[6.2682576179504395]], [[6.8438286781311035]], [[6.099343299865723]], [[6.04555082321167]], [[6.758257865905762]], [[7.5189290046691895]], [[6.542966842651367]], [[6.613718032836914]], [[5.846857070922852]], [[5.824804306030273]], [[6.4262542724609375]], [[6.430300235748291]], [[6.610617637634277]], [[6.656274795532227]], [[6.237132549285889]], [[5.95920991897583]], [[6.213052749633789]], [[6.1755523681640625]]]], dtype='float32').reshape([1, 24, 1, 1]),
        ]


class TestPrimitiveOp_2308dc4c012a75634d23dd4b43fc5c16(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1c21e9be9a7e71dc79ebb27483a54b38
    def get_inputs(self):
        return [
            paddle.uniform([145, 84], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_96f14ea372dec188e39c5fd3cee9ab19(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 28, 40], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6b9c6e072b3f4c87fd7393dc3ce44e7a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[1.498032808303833]], [[1.1699885129928589]], [[1.4005337953567505]], [[1.0293478965759277]]]], dtype='float32').reshape([1, 4, 1, 1]),
        ]


class TestPrimitiveOp_2308dc4c012a75634d23dd4b43fc5c16(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1c21e9be9a7e71dc79ebb27483a54b38
    def get_inputs(self):
        return [
            paddle.uniform([145, 84], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_bce2d7f600dca69abaac344acc4731d4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[2.946103572845459]], [[3.4548959732055664]], [[3.2269513607025146]], [[3.368286609649658]], [[2.8261561393737793]], [[2.9311068058013916]], [[2.6655144691467285]], [[2.6224160194396973]], [[3.169365167617798]], [[2.677142858505249]], [[3.360309600830078]]]], dtype='float32').reshape([1, 11, 1, 1]),
        ]


class TestPrimitiveOp_823c928d12a943ccf0179ecff50ea704(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.uniform([1, 40, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d43e57aa76dae90b491308a51147aeea(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.uniform([1, 240, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_08c0b2f2608ee187574a700caa4f0ce0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_99d2fe20e4bc583333c61fdc320bc356(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[8.565597534179688]], [[8.85666561126709]], [[7.50863790512085]], [[8.139659881591797]], [[7.596431255340576]], [[7.8915324211120605]], [[8.013626098632812]], [[8.710958480834961]], [[8.468538284301758]], [[7.645221710205078]], [[9.079865455627441]], [[8.134474754333496]], [[9.158711433410645]], [[8.963510513305664]], [[7.479450702667236]], [[8.753923416137695]], [[7.598970413208008]], [[7.580456256866455]], [[8.508934020996094]], [[8.746679306030273]], [[8.63934326171875]], [[7.524258613586426]], [[7.697452068328857]], [[8.258296012878418]], [[8.279424667358398]], [[7.730199337005615]], [[8.538504600524902]], [[8.165992736816406]], [[8.467819213867188]], [[7.427496910095215]]]], dtype='float32').reshape([1, 30, 1, 1]),
        ]


class TestPrimitiveOp_fb8f67f9e9c1678f4f4c8b0e74def09f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.uniform([1, 168, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f38364902031ddadc9585f62418c0cda(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1c21e9be9a7e71dc79ebb27483a54b38
    def get_inputs(self):
        return [
            paddle.uniform([145, 60], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e0cba9380cef8f69a9edc9687c0433af(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 80, 128], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_5a4fd0e79c249d5331092b1cd88cfddf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[4.462280750274658]], [[4.655839920043945]], [[4.034661769866943]], [[4.623201370239258]], [[4.026586532592773]], [[4.085882663726807]], [[4.6295390129089355]], [[4.664148807525635]], [[4.463987350463867]], [[4.678213119506836]], [[4.240274429321289]], [[4.858091354370117]], [[5.04475212097168]], [[3.7081854343414307]], [[3.8651325702667236]], [[4.311214447021484]]]], dtype='float32').reshape([1, 16, 1, 1]),
        ]


class TestPrimitiveOp_5969afac55b42e339678e6c2859eff99(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 14, 20], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_cb8ce128f385bbdadf232989019655d6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 22, 33], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6b259d06d95dbd06c41278a9f32c232d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 23, 35], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_7bbe2693465cdbfb6dab8dd2eaef4b87(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 46, 70], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_fb8f67f9e9c1678f4f4c8b0e74def09f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.uniform([1, 168, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e71e5a746f257adc03bb6a079cc8527f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1c21e9be9a7e71dc79ebb27483a54b38
    def get_inputs(self):
        return [
            paddle.uniform([22, 15], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_02d792bb15375f198f94661f3b17b2e2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_08c0b2f2608ee187574a700caa4f0ce0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_883951aa43eac171df7aed69d8fe0f3b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[7.979545593261719]], [[7.502386569976807]], [[7.407017707824707]], [[7.328598976135254]], [[7.504692554473877]], [[7.502708435058594]], [[7.600711822509766]], [[7.319614887237549]], [[7.540493488311768]], [[7.206352233886719]], [[7.858943462371826]], [[7.72793436050415]], [[8.392699241638184]], [[7.311511993408203]], [[7.200669765472412]], [[7.192445278167725]], [[6.876770496368408]], [[6.432420253753662]], [[7.926684856414795]], [[8.070440292358398]], [[6.884799480438232]], [[6.536991119384766]], [[7.266120910644531]], [[8.063619613647461]], [[7.6492228507995605]], [[7.3653388023376465]], [[7.466222763061523]], [[7.257701396942139]], [[7.193363189697266]], [[8.130621910095215]]]], dtype='float32').reshape([1, 30, 1, 1]),
        ]


class TestPrimitiveOp_c25415ea9777702401ba4be3fa415156(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_2f5a855286a2c426885c547829b42c8e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1c21e9be9a7e71dc79ebb27483a54b38
    def get_inputs(self):
        return [
            paddle.uniform([1, 218], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_cc6ff7aee76c67859ba983e5cf2c2dfd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[5.4261298179626465]], [[5.420971870422363]], [[5.311474323272705]], [[5.697546005249023]], [[5.705629825592041]], [[5.828099727630615]], [[5.449417591094971]], [[5.936191558837891]], [[5.457039833068848]], [[5.1325531005859375]], [[5.356573581695557]], [[6.186561584472656]], [[5.268749237060547]], [[6.0112738609313965]], [[5.64337158203125]], [[5.816868305206299]], [[6.055903434753418]], [[6.368608474731445]], [[5.593043804168701]], [[5.533454418182373]], [[6.144598484039307]], [[5.9172139167785645]], [[5.942316055297852]], [[5.2752885818481445]], [[5.371233940124512]]]], dtype='float32').reshape([1, 25, 1, 1]),
        ]


class TestPrimitiveOp_d43e57aa76dae90b491308a51147aeea(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.uniform([1, 240, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_0fd0cfa2d3b0bb77009d5d25322469e6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 6, 9], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_fb8f67f9e9c1678f4f4c8b0e74def09f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.uniform([1, 168, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f07a9167231b614dc088b834d8dfd0fd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a8dcc2522615a9fddf227cf60536fbbe(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1c21e9be9a7e71dc79ebb27483a54b38
    def get_inputs(self):
        return [
            paddle.uniform([390, 1024], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a8dcc2522615a9fddf227cf60536fbbe(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1c21e9be9a7e71dc79ebb27483a54b38
    def get_inputs(self):
        return [
            paddle.uniform([390, 1024], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4eb3813fdf586851a882ada3ce7f8a8c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1c21e9be9a7e71dc79ebb27483a54b38
    def get_inputs(self):
        return [
            paddle.uniform([171, 84], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3f6f6e508c4a4a5d388ae6d028393736(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.uniform([1, 60, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_cd8a6c4e4170e39896748686865f8b68(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[4.819231986999512]], [[4.967597007751465]], [[5.137096881866455]], [[4.944881439208984]], [[5.179754257202148]], [[4.973467826843262]], [[4.452411651611328]], [[4.99772310256958]], [[5.283167362213135]], [[4.811030387878418]], [[4.926609516143799]], [[4.973180770874023]], [[4.5081257820129395]], [[4.762726306915283]], [[5.076358318328857]], [[5.03634786605835]], [[5.133474349975586]], [[4.7684645652771]], [[4.720879554748535]], [[4.631831169128418]]]], dtype='float32').reshape([1, 20, 1, 1]),
        ]


class TestPrimitiveOp_fb8f67f9e9c1678f4f4c8b0e74def09f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.uniform([1, 168, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c25415ea9777702401ba4be3fa415156(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_08c0b2f2608ee187574a700caa4f0ce0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_fb8f67f9e9c1678f4f4c8b0e74def09f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.uniform([1, 168, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_fa7cbdf27ba30d086d81cb908dbe79ec(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[3.8294167518615723]], [[4.928394794464111]], [[5.4254655838012695]], [[5.18821907043457]], [[4.075259685516357]], [[5.00288724899292]], [[5.440080642700195]], [[5.044766426086426]], [[4.88026237487793]], [[5.161048412322998]], [[4.695656776428223]], [[4.971778869628906]], [[4.89567232131958]], [[5.618208885192871]], [[5.1241254806518555]], [[5.2426371574401855]], [[4.9540934562683105]], [[4.764132499694824]]]], dtype='float32').reshape([1, 18, 1, 1]),
        ]


class TestPrimitiveOp_a74c36a06b1620d2457fa90aab055645(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1c21e9be9a7e71dc79ebb27483a54b38
    def get_inputs(self):
        return [
            paddle.uniform([10, 60], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c806fc34a2e4e2958d32485bf1c44917(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.uniform([1, 144, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_cf284e16d712aa583b185cae6271c24a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 7, 10], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_fb8f67f9e9c1678f4f4c8b0e74def09f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.uniform([1, 168, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_bc2fb0ac525f1fb60f7c193687596e6b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.uniform([43, 96, 109, 109], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9c811429d58905a3e38e7e2ccffc90e2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.uniform([43, 16, 54, 54], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_55672c9d54e59ccdb88652896e9abc80(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.uniform([43, 64, 54, 54], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_55672c9d54e59ccdb88652896e9abc80(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.uniform([43, 64, 54, 54], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9c811429d58905a3e38e7e2ccffc90e2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.uniform([43, 16, 54, 54], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_55672c9d54e59ccdb88652896e9abc80(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.uniform([43, 64, 54, 54], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_55672c9d54e59ccdb88652896e9abc80(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.uniform([43, 64, 54, 54], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_2a82319cc7128d1f53fec39b8a2bcfc8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.uniform([43, 32, 54, 54], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e4d0051213263df7bbc3c5bb6f9bff20(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.uniform([43, 128, 54, 54], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e4d0051213263df7bbc3c5bb6f9bff20(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.uniform([43, 128, 54, 54], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_63eab2e222abadb41ce71d40294ebadf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.uniform([43, 32, 26, 26], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_47eda4bbfcb222c0e061db87a3ee6914(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.uniform([43, 128, 26, 26], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_47eda4bbfcb222c0e061db87a3ee6914(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.uniform([43, 128, 26, 26], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b92f88bab971c814f2a270c1fb5ae4b9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.uniform([43, 48, 26, 26], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c3ba7d3785eac1ff1ecf74178180b6d8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.uniform([43, 192, 26, 26], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c3ba7d3785eac1ff1ecf74178180b6d8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.uniform([43, 192, 26, 26], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b92f88bab971c814f2a270c1fb5ae4b9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.uniform([43, 48, 26, 26], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c3ba7d3785eac1ff1ecf74178180b6d8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.uniform([43, 192, 26, 26], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c3ba7d3785eac1ff1ecf74178180b6d8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.uniform([43, 192, 26, 26], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_91d64d987b3cdf9f62ef83c7149369b8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.uniform([43, 64, 26, 26], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_5545abf22552e38ef11ac5ffeb488c51(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.uniform([43, 256, 26, 26], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_5545abf22552e38ef11ac5ffeb488c51(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.uniform([43, 256, 26, 26], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_31de18d0ac50122c03743710a908faf5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.uniform([43, 64, 12, 12], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ef58ef18d7feb57b8740f8b5ef4d05d1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.uniform([43, 256, 12, 12], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ef58ef18d7feb57b8740f8b5ef4d05d1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.uniform([43, 256, 12, 12], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9e94cce4ba38721019b282ea13622fd6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.uniform([43, 1000, 12, 12], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3f6f6e508c4a4a5d388ae6d028393736(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.uniform([1, 60, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f53b09bf2f637e4ac85637e4545c83a3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[4.579583644866943]], [[3.987138032913208]], [[4.566008567810059]], [[5.609007835388184]], [[4.824344158172607]], [[4.967480659484863]], [[4.597592353820801]], [[4.221213340759277]], [[4.373418807983398]], [[4.443808555603027]], [[3.9890925884246826]], [[4.554945468902588]], [[4.303409099578857]], [[5.069254398345947]], [[4.908219337463379]], [[4.3355712890625]], [[4.356297016143799]], [[4.193089008331299]]]], dtype='float32').reshape([1, 18, 1, 1]),
        ]


class TestPrimitiveOp_cec624690c57e95be692914c3a76b07a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1c21e9be9a7e71dc79ebb27483a54b38
    def get_inputs(self):
        return [
            paddle.uniform([10, 84], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_90c394e12b7af3cc362171332fb3c955(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[6.418142318725586]], [[5.7315592765808105]], [[6.051327705383301]], [[6.1229119300842285]], [[5.347337245941162]], [[5.3859968185424805]], [[6.045626163482666]], [[5.939547061920166]], [[6.08483362197876]], [[6.583146572113037]], [[5.6943359375]], [[6.144874095916748]], [[6.776207447052002]], [[6.042275428771973]], [[5.6396894454956055]], [[5.307939052581787]], [[7.0514750480651855]], [[6.645849704742432]], [[5.006920337677002]], [[5.742197036743164]], [[6.737719535827637]], [[6.357386589050293]], [[6.788431167602539]], [[6.193100929260254]]]], dtype='float32').reshape([1, 24, 1, 1]),
        ]


class TestPrimitiveOp_e97138f5dcb67670e56dccdcab817829(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 11, 17], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e1e3dc614740a07ba5bbb365b45356a5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[4.553216934204102]], [[5.243834972381592]], [[5.057967662811279]], [[4.7062458992004395]], [[5.247849941253662]], [[5.114867210388184]], [[4.990291118621826]], [[4.798624515533447]], [[5.134045600891113]], [[4.944972515106201]], [[4.478386402130127]], [[5.6592254638671875]], [[5.583248615264893]], [[5.141046047210693]], [[5.647624492645264]], [[4.361642360687256]], [[5.095796585083008]], [[4.453990936279297]]]], dtype='float32').reshape([1, 18, 1, 1]),
        ]


class TestPrimitiveOp_1450dcd5a31f05d77082a2188cf7e55d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.uniform([1, 48, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_61ed39edc6471f6d5e1cdee4e00dc9c0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 10, 16], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_2babc24a0310a35c911c5f0eb4d96bfe(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[4.06638240814209]], [[5.281164646148682]], [[4.484489440917969]], [[4.670993328094482]], [[4.920041084289551]], [[4.694016933441162]], [[4.3606858253479]], [[4.4463958740234375]], [[5.016234874725342]], [[4.9394707679748535]], [[4.267251014709473]], [[5.512999534606934]], [[4.754570007324219]], [[4.967658996582031]], [[4.833072185516357]], [[5.408027648925781]], [[4.468790054321289]], [[5.006646633148193]]]], dtype='float32').reshape([1, 18, 1, 1]),
        ]


class TestPrimitiveOp_1450dcd5a31f05d77082a2188cf7e55d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.uniform([1, 48, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_5a7f4353800528954beaf243948f6705(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1c21e9be9a7e71dc79ebb27483a54b38
    def get_inputs(self):
        return [
            paddle.uniform([10, 9], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_bf80cb435da63cf74c986a092fe8a699(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 12, 18], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_cd54e934e10931aa72206219f179b00c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.uniform([10, 96, 109, 109], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d7cb24c681fc67c6e544e803dabb266a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.uniform([10, 16, 54, 54], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_bfcccaf928df930747c0b7b366baa1e4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.uniform([10, 64, 54, 54], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_bfcccaf928df930747c0b7b366baa1e4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.uniform([10, 64, 54, 54], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d7cb24c681fc67c6e544e803dabb266a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.uniform([10, 16, 54, 54], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_bfcccaf928df930747c0b7b366baa1e4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.uniform([10, 64, 54, 54], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_bfcccaf928df930747c0b7b366baa1e4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.uniform([10, 64, 54, 54], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_cc1599784854eb38da665f992ffffc7c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.uniform([10, 32, 54, 54], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_8b2b501677486870aebd6cab9ac6e437(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.uniform([10, 128, 54, 54], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_8b2b501677486870aebd6cab9ac6e437(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.uniform([10, 128, 54, 54], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_7d1bf20a32a22c4790aa610efa6b8734(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.uniform([10, 32, 26, 26], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3ca4399b6e6fddd98ff451800d8f644c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.uniform([10, 128, 26, 26], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3ca4399b6e6fddd98ff451800d8f644c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.uniform([10, 128, 26, 26], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_152e9f4617287e451b2b3eb90b7fcbd5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.uniform([10, 48, 26, 26], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_8301c7021fe7eb4914afa2f13eb106bb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.uniform([10, 192, 26, 26], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_8301c7021fe7eb4914afa2f13eb106bb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.uniform([10, 192, 26, 26], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_152e9f4617287e451b2b3eb90b7fcbd5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.uniform([10, 48, 26, 26], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_8301c7021fe7eb4914afa2f13eb106bb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.uniform([10, 192, 26, 26], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_8301c7021fe7eb4914afa2f13eb106bb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.uniform([10, 192, 26, 26], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_19da6fc455678bd53641dcb0fabfaf4e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.uniform([10, 64, 26, 26], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d59e2eca5002651fc8b52b4064af77d3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.uniform([10, 256, 26, 26], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d59e2eca5002651fc8b52b4064af77d3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.uniform([10, 256, 26, 26], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b088caa91944cccdae1886b36925fc97(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.uniform([10, 64, 12, 12], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_acd363209df46d476ac4975cf4f11c7e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.uniform([10, 256, 12, 12], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_acd363209df46d476ac4975cf4f11c7e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.uniform([10, 256, 12, 12], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c2ed01e0efdfc42daa61ab025da48cbf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.uniform([10, 1000, 12, 12], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_59fa089f133b648085a819887eed2d8a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1c21e9be9a7e71dc79ebb27483a54b38
    def get_inputs(self):
        return [
            paddle.uniform([10, 120], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_02d792bb15375f198f94661f3b17b2e2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6482c5c11535e8e9a9540f2a432f08c6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1c21e9be9a7e71dc79ebb27483a54b38
    def get_inputs(self):
        return [
            paddle.uniform([22, 84], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_7e20f3022c4878ecf88f16b46ca5950d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 92, 140], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_90c0e70cca7060e6be983d978f5cc194(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 12, 18], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_830cdb097ab410421892dc77cc331372(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1c21e9be9a7e71dc79ebb27483a54b38
    def get_inputs(self):
        return [
            paddle.uniform([171, 60], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4eb3813fdf586851a882ada3ce7f8a8c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1c21e9be9a7e71dc79ebb27483a54b38
    def get_inputs(self):
        return [
            paddle.uniform([171, 84], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a97513177df8d1f0fb07e17f3a5922a6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 300, 300], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a97513177df8d1f0fb07e17f3a5922a6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 300, 300], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_24bf35cc19e07d813738034948d9cf00(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 150, 150], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_24bf35cc19e07d813738034948d9cf00(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 150, 150], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9b7fdf4b931cfddc6861c124f2d7d359(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 75, 75], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9b7fdf4b931cfddc6861c124f2d7d359(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 75, 75], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9b7fdf4b931cfddc6861c124f2d7d359(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 75, 75], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_53d3d82922ab1ceb3f78313da0801a71(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 38, 38], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_53d3d82922ab1ceb3f78313da0801a71(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 38, 38], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_53d3d82922ab1ceb3f78313da0801a71(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 38, 38], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_919f9d2d2f600205d212ef654e8616cd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 19, 19], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_919f9d2d2f600205d212ef654e8616cd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 19, 19], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_919f9d2d2f600205d212ef654e8616cd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 19, 19], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d2591649927a5683f77fb176357e8536(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.uniform([1, 1024, 19, 19], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d2591649927a5683f77fb176357e8536(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.uniform([1, 1024, 19, 19], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_8f9dd67da782b70a78feadbf75bc7fe5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 19, 19], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b47b12762cda7529a5e63b917e6aaf0a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 10, 10], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_2816e0f9aca0264e0785f0ab8ae52492(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 10, 10], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_07217b095db6b5973aee3fedb43616c2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 5, 5], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4c1350fec15fb7975e4ee4dcd7fe612e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 5, 5], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c25a7c2d9c3a085826680f054f8ff821(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_1085c0ee48e00b9f791eb886a7555930(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c24728f3e017c1e2179f201537da8c32(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_fb8f67f9e9c1678f4f4c8b0e74def09f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.uniform([1, 168, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9f39e416a3ce42c7f17e3c2781828f65(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 13, 19], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_76a924819cdd9ba4a1d4852cbdecce55(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1c21e9be9a7e71dc79ebb27483a54b38
    def get_inputs(self):
        return [
            paddle.uniform([171, 15], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b42e4d5e628c97cf9e06ec2529b6429f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1c21e9be9a7e71dc79ebb27483a54b38
    def get_inputs(self):
        return [
            paddle.uniform([22, 60], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ec2949e2eff82edf6973343091d41c41(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 25, 38], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c26340617df277435d70aa5e0ad71497(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[4.6244988441467285]], [[4.329029560089111]], [[4.995797157287598]], [[4.316667079925537]], [[4.923176288604736]], [[4.73710298538208]], [[5.304345607757568]], [[4.629157066345215]], [[4.705936431884766]], [[4.469086170196533]], [[4.268731594085693]], [[4.755530834197998]], [[4.751047134399414]], [[4.774821758270264]], [[5.1132731437683105]], [[4.930696487426758]], [[5.313457012176514]], [[5.625310897827148]]]], dtype='float32').reshape([1, 18, 1, 1]),
        ]


class TestPrimitiveOp_3f6f6e508c4a4a5d388ae6d028393736(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.uniform([1, 60, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_aeadc7d43674f3d545121ba06d48ef89(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 13, 19], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_823c928d12a943ccf0179ecff50ea704(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.uniform([1, 40, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_7baad626bd61825710109e0129b29a31(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 16, 16], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f4e45a7f01ba76cbcfee5932086ea48c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[4.71750020980835]], [[3.766853094100952]], [[4.769667625427246]], [[4.383036136627197]], [[4.173023223876953]], [[3.5847434997558594]], [[4.281980991363525]], [[4.766328811645508]], [[4.207974433898926]], [[4.099235534667969]], [[4.517120838165283]], [[3.9880950450897217]], [[4.008715629577637]], [[3.37216854095459]], [[4.127290725708008]], [[3.942164659500122]]]], dtype='float32').reshape([1, 16, 1, 1]),
        ]


class TestPrimitiveOp_074219841dc2af73e92eabaf2ff4f706(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1c21e9be9a7e71dc79ebb27483a54b38
    def get_inputs(self):
        return [
            paddle.uniform([22, 9], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_08c0b2f2608ee187574a700caa4f0ce0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_332219a56b15cf64dae1c06805bb4a13(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[4.8236403465271]], [[5.052951335906982]], [[4.7229461669921875]], [[5.1916890144348145]], [[5.195699691772461]], [[4.67725133895874]], [[4.74697732925415]], [[5.135109901428223]], [[5.14848518371582]], [[5.57559061050415]], [[4.84277868270874]], [[4.417875289916992]], [[5.214022159576416]], [[5.045286655426025]], [[4.8187713623046875]], [[5.191462993621826]], [[5.3586812019348145]], [[4.862154006958008]]]], dtype='float32').reshape([1, 18, 1, 1]),
        ]


class TestPrimitiveOp_d80fb468cd5143859929e9d1eefdf93b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[1.9138317108154297]], [[1.692276120185852]], [[1.8661717176437378]], [[1.6739617586135864]]]], dtype='float32').reshape([1, 4, 1, 1]),
        ]


class TestPrimitiveOp_54bbc570c82e5176d4f66d1ed31d7f6c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.uniform([11, 96, 109, 109], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_2784442923530739fba18787af300a9b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.uniform([11, 16, 54, 54], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f0cb5f9de390ce2647292326589025a4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.uniform([11, 64, 54, 54], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f0cb5f9de390ce2647292326589025a4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.uniform([11, 64, 54, 54], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_2784442923530739fba18787af300a9b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.uniform([11, 16, 54, 54], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f0cb5f9de390ce2647292326589025a4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.uniform([11, 64, 54, 54], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f0cb5f9de390ce2647292326589025a4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.uniform([11, 64, 54, 54], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e9fc4575b2f9a76c5c862fb65dca8241(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.uniform([11, 32, 54, 54], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4d2db91cf500db2b417bf749f4a8ce47(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.uniform([11, 128, 54, 54], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4d2db91cf500db2b417bf749f4a8ce47(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.uniform([11, 128, 54, 54], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_1ae89960be8a1d72d3394fadc48a7fb1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.uniform([11, 32, 26, 26], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_45af406070a0785b6ee3b4a8c60c1c2b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.uniform([11, 128, 26, 26], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_45af406070a0785b6ee3b4a8c60c1c2b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.uniform([11, 128, 26, 26], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_2b6190c3657f3399fc64f4879578b792(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.uniform([11, 48, 26, 26], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6ce6892645457535bde306d496a9be18(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.uniform([11, 192, 26, 26], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6ce6892645457535bde306d496a9be18(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.uniform([11, 192, 26, 26], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_2b6190c3657f3399fc64f4879578b792(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.uniform([11, 48, 26, 26], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6ce6892645457535bde306d496a9be18(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.uniform([11, 192, 26, 26], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6ce6892645457535bde306d496a9be18(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.uniform([11, 192, 26, 26], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_58ceb9ffef24a09856c62737e7eb810f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.uniform([11, 64, 26, 26], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_895b6cc27e41218d16582c6289efa808(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.uniform([11, 256, 26, 26], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_895b6cc27e41218d16582c6289efa808(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.uniform([11, 256, 26, 26], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_7f6a8cce5a1e9b6fb3a9c9cfc829f5a9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.uniform([11, 64, 12, 12], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d051e139bd583a0928530ecedd2b77f1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.uniform([11, 256, 12, 12], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d051e139bd583a0928530ecedd2b77f1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.uniform([11, 256, 12, 12], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_235b323ef0be54ea27f5bb715d0ba747(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.uniform([11, 1000, 12, 12], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3f6f6e508c4a4a5d388ae6d028393736(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.uniform([1, 60, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f07a9167231b614dc088b834d8dfd0fd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ed370699ad64cf93fa397abb9096b602(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1c21e9be9a7e71dc79ebb27483a54b38
    def get_inputs(self):
        return [
            paddle.uniform([145, 15], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6d8de49024675fbdefcb22d8478d70f3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1c21e9be9a7e71dc79ebb27483a54b38
    def get_inputs(self):
        return [
            paddle.uniform([1, 168], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f8d2054b89c41285da2df8cca49057bf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.uniform([1, 100, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6482c5c11535e8e9a9540f2a432f08c6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1c21e9be9a7e71dc79ebb27483a54b38
    def get_inputs(self):
        return [
            paddle.uniform([22, 84], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3f6f6e508c4a4a5d388ae6d028393736(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.uniform([1, 60, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_08c0b2f2608ee187574a700caa4f0ce0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_1450dcd5a31f05d77082a2188cf7e55d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.uniform([1, 48, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_fb8f67f9e9c1678f4f4c8b0e74def09f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.uniform([1, 168, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_51bb5a8dea1746a520cc1655e6dba253(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[5.320950031280518]], [[5.029046058654785]], [[5.069671154022217]], [[4.538184642791748]], [[4.629547595977783]], [[4.70802640914917]], [[4.715033531188965]], [[4.512894630432129]], [[4.810446739196777]], [[4.68913459777832]], [[5.208148002624512]], [[5.160460948944092]], [[5.343917369842529]], [[4.764742374420166]], [[5.029192924499512]], [[5.373773097991943]], [[5.740528106689453]], [[4.85421085357666]], [[4.823370933532715]], [[5.062239170074463]]]], dtype='float32').reshape([1, 20, 1, 1]),
        ]


class TestPrimitiveOp_b4b88fda1d68de0fa5cd6205016818c3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.uniform([1, 84, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9d3779b70f7d6bbe6278676fec6a7330(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[3.601583242416382]], [[3.2522401809692383]], [[3.9426109790802]], [[2.8303322792053223]], [[3.6804349422454834]], [[3.1349878311157227]], [[3.369419813156128]], [[2.978642463684082]], [[3.2317445278167725]], [[3.1392486095428467]], [[3.2146658897399902]], [[3.248555898666382]]]], dtype='float32').reshape([1, 12, 1, 1]),
        ]


class TestPrimitiveOp_94dd3c7b74b526c8f50bca1b6af3223f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[4.887347221374512]], [[5.173083782196045]], [[5.676054954528809]], [[5.187004566192627]], [[5.235259532928467]], [[5.570391654968262]], [[5.4706244468688965]], [[4.995048522949219]], [[5.354491710662842]], [[5.069897651672363]], [[5.129307746887207]], [[5.034744739532471]], [[5.100893020629883]], [[5.21192741394043]], [[4.591923713684082]], [[5.437050819396973]], [[4.8461527824401855]], [[4.92936372756958]], [[4.811412811279297]], [[4.633780002593994]]]], dtype='float32').reshape([1, 20, 1, 1]),
        ]


class TestPrimitiveOp_9a411a11195e37a5ad948375050db00a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[3.239675998687744]], [[3.233325481414795]], [[3.1844775676727295]], [[3.2806575298309326]], [[3.5525128841400146]], [[3.763702392578125]], [[3.6452975273132324]], [[2.8868260383605957]], [[3.655923366546631]], [[3.5652997493743896]], [[3.570770263671875]]]], dtype='float32').reshape([1, 11, 1, 1]),
        ]


class TestPrimitiveOp_08c0b2f2608ee187574a700caa4f0ce0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_90c0e70cca7060e6be983d978f5cc194(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 12, 18], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f8d2054b89c41285da2df8cca49057bf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.uniform([1, 100, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f067cae410c989cfabfde3468af41020(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 56, 80], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_557cdb0f1d7cc63a82f1b2d1ede07eaf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[3.8458921909332275]], [[3.893167018890381]], [[4.0064873695373535]], [[4.077722549438477]], [[4.0647101402282715]], [[4.12076473236084]], [[4.0326828956604]], [[4.035554885864258]], [[3.636183977127075]], [[3.9118173122406006]], [[4.488284111022949]], [[3.9961748123168945]], [[3.807126760482788]], [[3.8151307106018066]]]], dtype='float32').reshape([1, 14, 1, 1]),
        ]


class TestPrimitiveOp_99fd0a874d9e0d574f00d31750f5af2f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_38f2fd615db526e352bb46393710f73a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 13, 19], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_823c928d12a943ccf0179ecff50ea704(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.uniform([1, 40, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b0611e6d9a7ae6f4f6ea090c4c3b0912(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[5.097085952758789]], [[6.046693325042725]], [[5.884105205535889]], [[4.725864410400391]], [[5.701006889343262]], [[5.194499969482422]], [[5.5775556564331055]], [[5.749197006225586]], [[5.571490287780762]], [[6.124773979187012]], [[6.355926036834717]], [[5.778453350067139]], [[5.824944496154785]], [[5.144975662231445]], [[6.082563877105713]], [[6.516267776489258]], [[6.008426666259766]], [[5.67205810546875]], [[5.723589897155762]], [[6.531248569488525]]]], dtype='float32').reshape([1, 20, 1, 1]),
        ]


class TestPrimitiveOp_86cc6a4cda63f9ff20d1ef156734ca90(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.uniform([2, 24, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_86cc6a4cda63f9ff20d1ef156734ca90(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.uniform([2, 24, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_86cc6a4cda63f9ff20d1ef156734ca90(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.uniform([2, 24, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_86cc6a4cda63f9ff20d1ef156734ca90(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.uniform([2, 24, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_cc9b2dbc551642cf922ae71d4650a54d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[34353.5546875]], [[38394.26953125]], [[34200.40234375]], [[31302.564453125]], [[43417.86328125]], [[30792.943359375]]], [[[34845.12109375]], [[38938.72265625]], [[34679.55859375]], [[31738.2265625]], [[44033.21484375]], [[31231.359375]]]], dtype='float32').reshape([2, 6, 1, 1]),
        ]


class TestPrimitiveOp_144fbd4de9a2f7960599cd69c67e180a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[42160.09765625]], [[43356.0234375]], [[34807.3203125]], [[41353.58984375]], [[43749.4765625]], [[40938.83203125]]], [[[40549.54296875]], [[41706.1171875]], [[33480.37890625]], [[39771.7265625]], [[42081.734375]], [[39377.5703125]]]], dtype='float32').reshape([2, 6, 1, 1]),
        ]


class TestPrimitiveOp_ee05e13d24a9879e2045c50ad579d3e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[37815.68359375]], [[46491.82421875]], [[37637.91796875]], [[36063.03125]], [[46331.71484375]], [[46617.09765625]]], [[[36356.3515625]], [[44704.7265625]], [[36185.62890625]], [[34671.5625]], [[44551.7734375]], [[44829.05859375]]]], dtype='float32').reshape([2, 6, 1, 1]),
        ]


class TestPrimitiveOp_1b88e79d6478ffaf998b340aa86553fa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[46384.93359375]], [[41454.01171875]], [[37989.71875]], [[40664.18359375]], [[45870.71484375]], [[38020.32421875]]], [[[44348.6796875]], [[39629.921875]], [[36315.76171875]], [[38875.1171875]], [[43854.77734375]], [[36348.61328125]]]], dtype='float32').reshape([2, 6, 1, 1]),
        ]


class TestPrimitiveOp_901a3ce812dc939dcfbabdb138ee5184(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 96, 144], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_901a3ce812dc939dcfbabdb138ee5184(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 96, 144], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_901a3ce812dc939dcfbabdb138ee5184(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 96, 144], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_901a3ce812dc939dcfbabdb138ee5184(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 96, 144], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_901a3ce812dc939dcfbabdb138ee5184(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 96, 144], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_901a3ce812dc939dcfbabdb138ee5184(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 96, 144], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_901a3ce812dc939dcfbabdb138ee5184(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 96, 144], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_901a3ce812dc939dcfbabdb138ee5184(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 96, 144], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_1c91308f01168ad9d9e89317d3d80c8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 48, 72], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_1c91308f01168ad9d9e89317d3d80c8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 48, 72], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_1c91308f01168ad9d9e89317d3d80c8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 48, 72], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_1c91308f01168ad9d9e89317d3d80c8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 48, 72], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_1c91308f01168ad9d9e89317d3d80c8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 48, 72], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_1c91308f01168ad9d9e89317d3d80c8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 48, 72], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_1c91308f01168ad9d9e89317d3d80c8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 48, 72], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_1c91308f01168ad9d9e89317d3d80c8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 48, 72], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e0271468a5b3b0bdb21cfbba05070a8b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 24, 36], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e0271468a5b3b0bdb21cfbba05070a8b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 24, 36], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e0271468a5b3b0bdb21cfbba05070a8b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 24, 36], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e0271468a5b3b0bdb21cfbba05070a8b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 24, 36], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e0271468a5b3b0bdb21cfbba05070a8b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 24, 36], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e0271468a5b3b0bdb21cfbba05070a8b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 24, 36], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e0271468a5b3b0bdb21cfbba05070a8b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 24, 36], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e0271468a5b3b0bdb21cfbba05070a8b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 24, 36], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_90c0e70cca7060e6be983d978f5cc194(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 12, 18], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_90c0e70cca7060e6be983d978f5cc194(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 12, 18], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_90c0e70cca7060e6be983d978f5cc194(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 12, 18], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_90c0e70cca7060e6be983d978f5cc194(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 12, 18], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_90c0e70cca7060e6be983d978f5cc194(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 12, 18], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_90c0e70cca7060e6be983d978f5cc194(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 12, 18], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_90c0e70cca7060e6be983d978f5cc194(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 12, 18], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_90c0e70cca7060e6be983d978f5cc194(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 12, 18], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3e7366fd4b7e963889c16cf09339f145(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 6, 9], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3e7366fd4b7e963889c16cf09339f145(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 6, 9], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3e7366fd4b7e963889c16cf09339f145(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 6, 9], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3e7366fd4b7e963889c16cf09339f145(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 6, 9], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3e7366fd4b7e963889c16cf09339f145(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 6, 9], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3e7366fd4b7e963889c16cf09339f145(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 6, 9], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3e7366fd4b7e963889c16cf09339f145(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 6, 9], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3e7366fd4b7e963889c16cf09339f145(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 6, 9], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_08c0b2f2608ee187574a700caa4f0ce0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_1450dcd5a31f05d77082a2188cf7e55d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.uniform([1, 48, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9b018e279113dc3e03c0ada053c0cf5c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[7.219970703125]], [[7.784126281738281]], [[7.827428817749023]], [[8.209232330322266]], [[7.692371845245361]], [[7.626750469207764]], [[8.033778190612793]], [[7.528597354888916]], [[7.796234130859375]], [[7.446854591369629]], [[7.121368885040283]], [[7.051188945770264]], [[7.789116859436035]], [[7.609338283538818]], [[6.6510443687438965]], [[7.625375747680664]], [[8.255825996398926]], [[8.235020637512207]], [[7.581562519073486]], [[7.541038990020752]], [[7.979496002197266]], [[7.4575347900390625]], [[7.476142406463623]], [[8.075897216796875]], [[7.466970443725586]], [[8.125590324401855]], [[7.947990894317627]], [[8.003846168518066]], [[7.388821601867676]], [[7.886233806610107]]]], dtype='float32').reshape([1, 30, 1, 1]),
        ]


class TestPrimitiveOp_9f4154fd4121673b1b331f82d087f2b7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[6.814724445343018]], [[6.774284362792969]], [[7.524858474731445]], [[6.822107791900635]], [[7.457672119140625]], [[7.345535755157471]], [[7.476398468017578]], [[7.1589579582214355]], [[7.322680950164795]], [[7.283619403839111]], [[7.031378746032715]], [[6.672905445098877]], [[6.4484171867370605]], [[7.1944966316223145]], [[7.280976295471191]], [[7.0551629066467285]], [[7.122607707977295]], [[7.416973114013672]], [[7.247720241546631]], [[6.785580635070801]], [[6.085677146911621]], [[6.414621353149414]], [[6.927490234375]], [[6.715959072113037]], [[6.866250038146973]], [[6.130156517028809]], [[6.598968505859375]], [[6.8014678955078125]], [[7.103934288024902]], [[7.52005672454834]]]], dtype='float32').reshape([1, 30, 1, 1]),
        ]


class TestPrimitiveOp_1e094fae8e2cdd7473b4e7bc248fadfa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 44, 66], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d3ea974c02d441446993b8bbdc7da59c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[8.571072578430176]], [[7.910855293273926]], [[7.639548301696777]], [[8.479670524597168]], [[9.00108814239502]], [[8.445219993591309]], [[7.997985363006592]], [[7.674211025238037]], [[8.481038093566895]], [[8.052382469177246]], [[8.052838325500488]], [[8.102134704589844]], [[9.469754219055176]], [[7.3469061851501465]], [[8.879927635192871]], [[7.816487789154053]], [[7.959301471710205]], [[8.373373985290527]], [[7.350678443908691]], [[8.218818664550781]], [[8.802421569824219]], [[7.801962852478027]], [[8.044658660888672]], [[8.281781196594238]], [[8.116142272949219]], [[8.111846923828125]], [[8.50810718536377]], [[9.174527168273926]], [[8.281989097595215]], [[7.9975972175598145]]]], dtype='float32').reshape([1, 30, 1, 1]),
        ]


class TestPrimitiveOp_5ec405c40243d91ca87fa1ed30386966(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.uniform([1, 50, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3f6f6e508c4a4a5d388ae6d028393736(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.uniform([1, 60, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_2302aac6e39f61ee0c91798c99c9c057(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[7.1592535972595215]], [[8.019697189331055]], [[9.132721900939941]], [[8.245113372802734]], [[8.201904296875]], [[8.027078628540039]], [[7.732528209686279]], [[8.434311866760254]], [[8.001360893249512]], [[8.272842407226562]], [[8.75783634185791]], [[8.043542861938477]], [[7.47999906539917]], [[8.129373550415039]], [[9.145278930664062]], [[7.5090179443359375]], [[7.697664737701416]], [[8.60932731628418]], [[6.7332634925842285]], [[8.00938892364502]], [[8.664624214172363]], [[8.28862476348877]], [[7.382583141326904]], [[8.051594734191895]], [[7.450173854827881]], [[8.692798614501953]], [[8.793678283691406]], [[8.495599746704102]], [[7.484212875366211]], [[9.080621719360352]]]], dtype='float32').reshape([1, 30, 1, 1]),
        ]


class TestPrimitiveOp_55a41ce45b8221c826c61b9a2b01ffe7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[4.03708553314209]], [[3.342827081680298]], [[3.163536548614502]], [[3.6480274200439453]], [[3.1267178058624268]], [[3.629101276397705]], [[3.069321870803833]], [[2.996713161468506]], [[2.89133882522583]], [[3.521888494491577]], [[3.979022979736328]], [[3.150074005126953]]]], dtype='float32').reshape([1, 12, 1, 1]),
        ]


class TestPrimitiveOp_57b52c4241a51b21f09b1975014adb16(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[2.799421787261963]], [[2.518261671066284]], [[2.8721373081207275]], [[2.9185006618499756]], [[2.6263015270233154]], [[2.7743587493896484]], [[3.625351667404175]], [[3.072920322418213]], [[2.7595460414886475]], [[2.802654981613159]], [[2.82415509223938]], [[3.3745367527008057]]]], dtype='float32').reshape([1, 12, 1, 1]),
        ]


class TestPrimitiveOp_a1476abc5973fae9ca0f2abfe094066c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[5.523202419281006]], [[6.377389430999756]], [[6.686032295227051]], [[5.554091453552246]], [[6.187918186187744]], [[5.89094877243042]], [[5.716607570648193]], [[6.461143493652344]], [[6.702357769012451]], [[6.33857536315918]], [[6.324361324310303]], [[6.188377857208252]], [[6.650539398193359]], [[6.015854835510254]], [[6.307999610900879]], [[6.638586044311523]], [[5.999324798583984]], [[6.248127460479736]], [[5.995881080627441]], [[5.776070594787598]], [[6.150759220123291]], [[7.176509380340576]], [[6.003063201904297]], [[6.2718658447265625]], [[6.431493282318115]]]], dtype='float32').reshape([1, 25, 1, 1]),
        ]


class TestPrimitiveOp_51d79759447e7eae9e133d320f3b2de0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.uniform([1, 72, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e32725dafd9fc3477272fa54d775ebef(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1c21e9be9a7e71dc79ebb27483a54b38
    def get_inputs(self):
        return [
            paddle.uniform([1, 312], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_50d711c6497e5aa743c20cf706ac454e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1c21e9be9a7e71dc79ebb27483a54b38
    def get_inputs(self):
        return [
            paddle.uniform([171, 120], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_04c5fe3a7b009c6daced35f3db7bd480(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1c21e9be9a7e71dc79ebb27483a54b38
    def get_inputs(self):
        return [
            paddle.uniform([145, 9], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_85f64dc53bf9ce25f5e0954718bbd0cf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 5, 8], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_8b93e19228e0bebe324baf623b567420(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[4.849146366119385]], [[5.562366485595703]], [[5.357488632202148]], [[4.600664138793945]], [[5.43474817276001]], [[5.1333842277526855]], [[4.804915904998779]], [[4.552052021026611]], [[4.770118236541748]], [[5.240309238433838]], [[4.955009937286377]], [[4.486055850982666]], [[4.7744245529174805]], [[5.813625335693359]], [[5.323536396026611]], [[4.844731330871582]], [[5.070008754730225]], [[4.528096675872803]]]], dtype='float32').reshape([1, 18, 1, 1]),
        ]


class TestPrimitiveOp_33f4a4436633b20f880732c633def918(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1c21e9be9a7e71dc79ebb27483a54b38
    def get_inputs(self):
        return [
            paddle.uniform([1, 39], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_0211dbb2d1f1b5df93e59f0791472d00(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[1.3530144691467285]], [[1.4647397994995117]], [[1.5248286724090576]], [[0.8225886821746826]], [[1.303797960281372]]]], dtype='float32').reshape([1, 5, 1, 1]),
        ]


class TestPrimitiveOp_1ed48ccfa0be0a8b53a4ad970134b36f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[3.0290560722351074]], [[2.792099714279175]], [[3.0667827129364014]], [[2.414238929748535]], [[2.9861397743225098]], [[3.1318516731262207]], [[3.273455858230591]], [[2.80271315574646]], [[2.6938159465789795]], [[2.9530866146087646]]]], dtype='float32').reshape([1, 10, 1, 1]),
        ]


class TestPrimitiveOp_c62519fddc1521756e87fced0f653a07(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[4.387393951416016]], [[5.253484725952148]], [[4.903465747833252]], [[5.175287246704102]], [[4.975003719329834]], [[4.971848487854004]], [[5.1791768074035645]], [[5.112245082855225]], [[5.15993595123291]], [[4.282576084136963]], [[5.401867389678955]], [[4.462133884429932]], [[5.145967483520508]], [[5.319954872131348]], [[5.04836893081665]], [[5.1379265785217285]], [[4.489475250244141]], [[5.0134477615356445]], [[4.9873480796813965]], [[5.273707866668701]]]], dtype='float32').reshape([1, 20, 1, 1]),
        ]


class TestPrimitiveOp_823c928d12a943ccf0179ecff50ea704(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.uniform([1, 40, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_5ec405c40243d91ca87fa1ed30386966(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.uniform([1, 50, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c25415ea9777702401ba4be3fa415156(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b3036a4887b77ccad8a52f82ca01261e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 10, 15], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_08c0b2f2608ee187574a700caa4f0ce0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_2f5a855286a2c426885c547829b42c8e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1c21e9be9a7e71dc79ebb27483a54b38
    def get_inputs(self):
        return [
            paddle.uniform([1, 218], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6d4c42d8b1c2475cd1432d27ed1d91de(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[6.206503868103027]], [[7.111752986907959]], [[6.231878280639648]], [[5.649900436401367]], [[7.215942859649658]], [[6.2238922119140625]], [[5.985706329345703]], [[6.53260612487793]], [[7.171493053436279]], [[6.2621378898620605]], [[6.22059965133667]], [[6.320767879486084]], [[6.109984874725342]], [[6.1065568923950195]], [[6.8695759773254395]], [[6.2865705490112305]], [[5.723404407501221]], [[6.720170974731445]], [[6.197271823883057]], [[6.079874038696289]], [[6.363616943359375]], [[6.1645965576171875]], [[5.554739475250244]], [[6.695564270019531]]]], dtype='float32').reshape([1, 24, 1, 1]),
        ]


class TestPrimitiveOp_801d2caad76869215f563adf18fc9f5c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1c21e9be9a7e71dc79ebb27483a54b38
    def get_inputs(self):
        return [
            paddle.uniform([22, 120], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_519a62aec3deb3849046230a33c20b14(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[2.351255178451538]], [[3.0869038105010986]], [[2.3032050132751465]], [[2.3017141819000244]], [[2.4560351371765137]], [[3.090928077697754]], [[2.6265499591827393]], [[2.695026159286499]], [[2.9914133548736572]], [[2.802804946899414]]]], dtype='float32').reshape([1, 10, 1, 1]),
        ]


class TestPrimitiveOp_0e4d13b091a1df9e33bb686df59e7c8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1c21e9be9a7e71dc79ebb27483a54b38
    def get_inputs(self):
        return [
            paddle.uniform([145, 120], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c001244499b147d8fba978ef5dd1513c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 40, 64], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_454825ea3bee908144a364fe43a63c8e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 50, 76], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c095c3f359b0b48e50f439bb41402681(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1c21e9be9a7e71dc79ebb27483a54b38
    def get_inputs(self):
        return [
            paddle.uniform([171, 9], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_08c0b2f2608ee187574a700caa4f0ce0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_412d0eb8cd63b7ba94978bb21fb7c023(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[4.880243301391602]], [[4.284125804901123]], [[5.1647257804870605]], [[5.034191608428955]], [[4.1328043937683105]], [[3.989349365234375]], [[4.830134868621826]], [[3.9868626594543457]], [[3.904547929763794]], [[4.827010154724121]], [[3.8705732822418213]], [[4.75415563583374]], [[3.9656474590301514]], [[4.086240768432617]], [[4.686644077301025]], [[4.369600296020508]], [[4.487377643585205]], [[4.246503829956055]]]], dtype='float32').reshape([1, 18, 1, 1]),
        ]


class TestPrimitiveOp_627e00b53617a841f51488df6dec73de(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1c21e9be9a7e71dc79ebb27483a54b38
    def get_inputs(self):
        return [
            paddle.to_tensor([[7.380273342132568, 6.993586540222168, 7.374476909637451, 7.281966686248779, 7.989966869354248, 8.266085624694824, 7.489744186401367, 8.130762100219727, 7.13471794128418, 7.360182285308838, 8.598377227783203, 7.828280925750732, 7.041106700897217, 7.472853660583496, 7.848850727081299, 8.005196571350098, 8.92200756072998, 8.074055671691895, 7.2128825187683105, 7.177803993225098, 8.775609016418457, 8.413022994995117, 7.756340026855469, 8.30650520324707, 7.593331813812256, 7.764378547668457, 7.462083339691162, 8.445094108581543, 7.404086589813232, 7.836207389831543]], dtype='float32').reshape([1, 30]),
        ]


class TestPrimitiveOp_fb8f67f9e9c1678f4f4c8b0e74def09f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.uniform([1, 168, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_fb8f67f9e9c1678f4f4c8b0e74def09f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.uniform([1, 168, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6d8de49024675fbdefcb22d8478d70f3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1c21e9be9a7e71dc79ebb27483a54b38
    def get_inputs(self):
        return [
            paddle.uniform([1, 168], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_44bb3df460b017e2b7992461a0c46bc6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[7.62192440032959]], [[7.400206089019775]], [[7.748739719390869]], [[7.875727653503418]], [[7.208639144897461]], [[7.998118877410889]], [[7.931631088256836]], [[7.8193678855896]], [[7.174508094787598]], [[8.173986434936523]], [[8.253493309020996]], [[7.992372989654541]], [[7.904558181762695]], [[7.437554836273193]], [[7.751658916473389]], [[8.34387493133545]], [[8.273571014404297]], [[7.511903285980225]], [[7.043776512145996]], [[8.035367965698242]], [[7.823747634887695]], [[7.011332035064697]], [[7.618360996246338]], [[7.738470077514648]], [[8.361632347106934]], [[7.775673866271973]], [[7.732045650482178]], [[8.275997161865234]], [[8.61551284790039]], [[7.694608211517334]]]], dtype='float32').reshape([1, 30, 1, 1]),
        ]


class TestPrimitiveOp_14f3d0d8a85bf3c3bf3ed1c1a8864a86(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[1.7743725776672363]], [[1.4237773418426514]], [[1.4726775884628296]], [[1.4478960037231445]], [[1.8577220439910889]]]], dtype='float32').reshape([1, 5, 1, 1]),
        ]


class TestPrimitiveOp_eac96c5abd515b52333b720654d0eacf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[2.6523306369781494]], [[2.4763336181640625]], [[3.2539005279541016]], [[2.871127128601074]], [[3.1752514839172363]], [[2.687026262283325]], [[2.6177079677581787]], [[2.592658042907715]], [[2.8259291648864746]], [[2.734337329864502]]]], dtype='float32').reshape([1, 10, 1, 1]),
        ]


class TestPrimitiveOp_aa0119103965971682315c59e25cb194(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[5.181726932525635]], [[5.7877678871154785]], [[6.224140644073486]], [[5.852336883544922]], [[4.836845874786377]], [[5.370150566101074]], [[5.751057147979736]], [[5.799452781677246]], [[6.094420433044434]], [[5.716272354125977]], [[5.1642937660217285]], [[5.500518798828125]], [[5.989058494567871]], [[4.902487277984619]], [[5.668533802032471]], [[5.383454322814941]], [[5.875471591949463]], [[5.620500564575195]], [[5.334749221801758]], [[5.4819560050964355]]]], dtype='float32').reshape([1, 20, 1, 1]),
        ]


class TestPrimitiveOp_823c928d12a943ccf0179ecff50ea704(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.uniform([1, 40, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_28c3a073e7f940f320ee39f6bc6afe4d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[3.760829210281372]], [[4.256175994873047]], [[4.550135612487793]], [[4.613018035888672]], [[5.112522125244141]], [[3.745678186416626]], [[4.642297267913818]], [[4.18134069442749]], [[3.7590386867523193]], [[4.86866569519043]], [[4.42974328994751]], [[3.9300332069396973]], [[4.057292461395264]], [[4.5839738845825195]], [[4.189048767089844]], [[3.859541416168213]]]], dtype='float32').reshape([1, 16, 1, 1]),
        ]


class TestPrimitiveOp_02d792bb15375f198f94661f3b17b2e2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_daa127595e1d8aa2c9dda19e7e02a7bb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.uniform([1, 36, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b6299460b133d186cd344e2cda109043(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 92, 140], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b6299460b133d186cd344e2cda109043(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 92, 140], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b6299460b133d186cd344e2cda109043(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 92, 140], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b6299460b133d186cd344e2cda109043(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 92, 140], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b6299460b133d186cd344e2cda109043(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 92, 140], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b6299460b133d186cd344e2cda109043(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 92, 140], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b6299460b133d186cd344e2cda109043(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 92, 140], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b6299460b133d186cd344e2cda109043(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 92, 140], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_cbf23431bf8a962b1124f1ded4a2bed4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 46, 70], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_cbf23431bf8a962b1124f1ded4a2bed4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 46, 70], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_cbf23431bf8a962b1124f1ded4a2bed4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 46, 70], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_cbf23431bf8a962b1124f1ded4a2bed4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 46, 70], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_cbf23431bf8a962b1124f1ded4a2bed4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 46, 70], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_cbf23431bf8a962b1124f1ded4a2bed4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 46, 70], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_cbf23431bf8a962b1124f1ded4a2bed4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 46, 70], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_cbf23431bf8a962b1124f1ded4a2bed4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 46, 70], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_8b14287a1bd76ada9b4aac1b9d3245ff(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 23, 35], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_8b14287a1bd76ada9b4aac1b9d3245ff(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 23, 35], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_8b14287a1bd76ada9b4aac1b9d3245ff(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 23, 35], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_8b14287a1bd76ada9b4aac1b9d3245ff(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 23, 35], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_8b14287a1bd76ada9b4aac1b9d3245ff(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 23, 35], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_8b14287a1bd76ada9b4aac1b9d3245ff(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 23, 35], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_8b14287a1bd76ada9b4aac1b9d3245ff(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 23, 35], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_8b14287a1bd76ada9b4aac1b9d3245ff(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 23, 35], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_90c0e70cca7060e6be983d978f5cc194(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 12, 18], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_90c0e70cca7060e6be983d978f5cc194(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 12, 18], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_90c0e70cca7060e6be983d978f5cc194(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 12, 18], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_90c0e70cca7060e6be983d978f5cc194(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 12, 18], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_90c0e70cca7060e6be983d978f5cc194(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 12, 18], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_90c0e70cca7060e6be983d978f5cc194(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 12, 18], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_90c0e70cca7060e6be983d978f5cc194(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 12, 18], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_90c0e70cca7060e6be983d978f5cc194(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 12, 18], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3e7366fd4b7e963889c16cf09339f145(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 6, 9], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3e7366fd4b7e963889c16cf09339f145(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 6, 9], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3e7366fd4b7e963889c16cf09339f145(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 6, 9], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3e7366fd4b7e963889c16cf09339f145(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 6, 9], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3e7366fd4b7e963889c16cf09339f145(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 6, 9], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3e7366fd4b7e963889c16cf09339f145(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 6, 9], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3e7366fd4b7e963889c16cf09339f145(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 6, 9], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3e7366fd4b7e963889c16cf09339f145(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 6, 9], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b4b88fda1d68de0fa5cd6205016818c3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.uniform([1, 84, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_99fd0a874d9e0d574f00d31750f5af2f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_62265257126109eb704e2f967976658f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[3.1301262378692627]], [[3.0191736221313477]], [[2.864104747772217]], [[3.18454909324646]], [[3.425471782684326]], [[3.2734270095825195]], [[3.5372228622436523]], [[3.3475706577301025]], [[3.380138635635376]], [[3.4204189777374268]], [[2.9306490421295166]], [[3.359321117401123]], [[2.9161040782928467]], [[2.9420838356018066]]]], dtype='float32').reshape([1, 14, 1, 1]),
        ]


class TestPrimitiveOp_5aa25b12109361433fdae29c293267f2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[5.636407852172852]], [[4.719157695770264]], [[4.447378635406494]], [[4.884665012359619]], [[4.541313171386719]], [[4.4184441566467285]], [[4.3862786293029785]], [[4.829842567443848]], [[4.4603729248046875]], [[4.298841953277588]], [[5.239867687225342]], [[4.0581583976745605]], [[4.3401641845703125]], [[4.505803108215332]], [[4.745649337768555]], [[5.160227298736572]], [[4.51344633102417]], [[4.922163009643555]], [[4.5199103355407715]], [[4.295330047607422]]]], dtype='float32').reshape([1, 20, 1, 1]),
        ]


class TestPrimitiveOp_25a451ded8d74b717b42b81c602cb4d5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 14, 20], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c25415ea9777702401ba4be3fa415156(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_0b5f691013d2963be72305c65f87bcf2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[7.801311492919922]], [[8.477471351623535]], [[7.692463397979736]], [[8.016641616821289]], [[7.590599060058594]], [[7.4950151443481445]], [[8.187170028686523]], [[8.017263412475586]], [[7.549017429351807]], [[7.662537574768066]], [[8.0874662399292]], [[8.07473087310791]], [[7.834475517272949]], [[8.067551612854004]], [[8.353202819824219]], [[8.677970886230469]], [[7.78714656829834]], [[8.194197654724121]], [[8.130566596984863]], [[7.409244537353516]], [[7.827935218811035]], [[8.826498031616211]], [[7.405838489532471]], [[8.205777168273926]], [[8.090394020080566]], [[8.595943450927734]], [[8.288196563720703]], [[7.462908744812012]], [[8.489489555358887]], [[7.518553733825684]]]], dtype='float32').reshape([1, 30, 1, 1]),
        ]


class TestPrimitiveOp_d43e57aa76dae90b491308a51147aeea(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.uniform([1, 240, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_823c928d12a943ccf0179ecff50ea704(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.uniform([1, 40, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_daa127595e1d8aa2c9dda19e7e02a7bb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.uniform([1, 36, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_83758a380378cdfcc70da7abf72a75b3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.uniform([22, 96, 109, 109], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a11724114270a53b3468edd3bfb54f84(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.uniform([22, 16, 54, 54], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_df61979c02220e953d40ecd6908ba351(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.uniform([22, 64, 54, 54], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_df61979c02220e953d40ecd6908ba351(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.uniform([22, 64, 54, 54], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a11724114270a53b3468edd3bfb54f84(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.uniform([22, 16, 54, 54], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_df61979c02220e953d40ecd6908ba351(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.uniform([22, 64, 54, 54], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_df61979c02220e953d40ecd6908ba351(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.uniform([22, 64, 54, 54], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_cbd9f859c24f0eb39a0427f15aea4bde(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.uniform([22, 32, 54, 54], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_fdb7a0d34c3fa06c2f95deafddb12dcc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.uniform([22, 128, 54, 54], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_fdb7a0d34c3fa06c2f95deafddb12dcc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.uniform([22, 128, 54, 54], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f2aaed711cdcd394fd1f928fb5730cd4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.uniform([22, 32, 26, 26], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_af1175f0f41617143f6a7dad25ece6bd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.uniform([22, 128, 26, 26], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_af1175f0f41617143f6a7dad25ece6bd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.uniform([22, 128, 26, 26], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c9a2828de00611a56089e794b3148cc9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.uniform([22, 48, 26, 26], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ba5307f9dbf14656f7586b81f88b61ef(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.uniform([22, 192, 26, 26], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ba5307f9dbf14656f7586b81f88b61ef(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.uniform([22, 192, 26, 26], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c9a2828de00611a56089e794b3148cc9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.uniform([22, 48, 26, 26], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ba5307f9dbf14656f7586b81f88b61ef(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.uniform([22, 192, 26, 26], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ba5307f9dbf14656f7586b81f88b61ef(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.uniform([22, 192, 26, 26], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_5e64b5153e5886f6766ad0ac3b7489e6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.uniform([22, 64, 26, 26], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a87308725fea72000840ae57c3764172(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.uniform([22, 256, 26, 26], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a87308725fea72000840ae57c3764172(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.uniform([22, 256, 26, 26], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_008417f752a3372e89d1ace26ee7a76f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.uniform([22, 64, 12, 12], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_da827879dbc66317fa7908b28d944337(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.uniform([22, 256, 12, 12], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_da827879dbc66317fa7908b28d944337(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.uniform([22, 256, 12, 12], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_721e31f8529304f90c330423f49b8325(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.uniform([22, 1000, 12, 12], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_5ec405c40243d91ca87fa1ed30386966(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.uniform([1, 50, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_8595935a8b519c766a6c3869080aefac(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 10, 16], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f8d2054b89c41285da2df8cca49057bf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.uniform([1, 100, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3e5feff90d442618740cc57bb654ea1a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[6.737256050109863]], [[6.288643836975098]], [[7.323916435241699]], [[6.758274555206299]], [[6.437505722045898]], [[6.254459381103516]], [[7.311783790588379]], [[6.879544734954834]], [[6.972203731536865]], [[7.086414813995361]], [[7.219585418701172]], [[6.86547327041626]], [[6.434556007385254]], [[6.952660083770752]], [[7.1039958000183105]], [[6.371720314025879]], [[6.167044639587402]], [[7.03317928314209]], [[6.506866455078125]], [[6.954500675201416]], [[6.295381546020508]], [[7.109879493713379]], [[6.929393768310547]], [[7.029480457305908]]]], dtype='float32').reshape([1, 24, 1, 1]),
        ]


class TestPrimitiveOp_00cdb00d4b40762c3209ffb58d208364(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[7.119134426116943]], [[5.923222541809082]], [[7.408734321594238]], [[6.47472620010376]], [[6.32647180557251]], [[6.711030960083008]], [[7.998645305633545]], [[6.787867069244385]], [[7.9734272956848145]], [[6.716749668121338]], [[7.480202674865723]], [[7.065846920013428]], [[7.417253017425537]], [[6.721028804779053]], [[6.85184907913208]], [[6.518846035003662]], [[7.006399631500244]], [[7.119091510772705]], [[7.460700988769531]], [[6.329360008239746]], [[6.411314964294434]], [[7.972550392150879]], [[7.204898357391357]], [[6.429723262786865]], [[7.317016124725342]]]], dtype='float32').reshape([1, 25, 1, 1]),
        ]


class TestPrimitiveOp_5d55dfd85308ab617e32c916ae76847e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[3.5534021854400635]], [[3.0459494590759277]], [[2.9850471019744873]], [[2.926222324371338]], [[3.026776075363159]], [[3.4852294921875]], [[3.641829490661621]], [[2.8090176582336426]], [[3.4768717288970947]], [[3.0273609161376953]], [[3.431544303894043]], [[3.369569778442383]]]], dtype='float32').reshape([1, 12, 1, 1]),
        ]


class TestPrimitiveOp_08c0b2f2608ee187574a700caa4f0ce0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_823c928d12a943ccf0179ecff50ea704(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.uniform([1, 40, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c25415ea9777702401ba4be3fa415156(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b3036a4887b77ccad8a52f82ca01261e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 10, 15], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_2cd4c5b8655dcfb401cc39534e808a67(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 25, 38], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_02d792bb15375f198f94661f3b17b2e2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_405d5a7003074597df339ef346f4ed1a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 112, 160], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_823c928d12a943ccf0179ecff50ea704(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.uniform([1, 40, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_08c0b2f2608ee187574a700caa4f0ce0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_32b0b4660d1e81a24da99a821c30a0b2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 7, 10], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_40ac5c623b0ccedb4d7c4f8992ad2de6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[762.2291259765625]], [[786.2742919921875]], [[743.0218505859375]], [[697.415283203125]], [[699.0284423828125]], [[712.4510498046875]], [[749.775634765625]], [[764.63330078125]], [[686.6480102539062]], [[747.8648681640625]], [[747.8525390625]], [[758.94287109375]], [[732.6702270507812]], [[676.7434692382812]], [[756.5175170898438]], [[697.097900390625]], [[741.1431884765625]], [[777.205078125]], [[683.3544311523438]], [[776.3590087890625]], [[679.1268310546875]], [[784.6321411132812]], [[701.3018798828125]], [[703.9038696289062]]]], dtype='float32').reshape([1, 24, 1, 1]),
        ]


class TestPrimitiveOp_407db019539ab5bdd4ffafcfa9a0acae(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[89.13827514648438]], [[99.29498291015625]], [[104.95667266845703]], [[104.80584716796875]], [[103.9608383178711]], [[85.3790054321289]], [[97.73500061035156]], [[90.42082977294922]], [[97.99455261230469]], [[106.14417266845703]], [[100.25833892822266]], [[103.96082305908203]], [[94.44489288330078]], [[91.15996551513672]], [[104.29300689697266]], [[100.65071868896484]], [[103.41301727294922]], [[109.82306671142578]], [[92.8393325805664]], [[106.42680358886719]], [[100.84012603759766]], [[99.505615234375]], [[94.85469818115234]], [[98.58260345458984]]]], dtype='float32').reshape([1, 24, 1, 1]),
        ]


class TestPrimitiveOp_749d3564745be60c625904793114604c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[47.50959014892578]], [[41.27824783325195]], [[42.36391830444336]], [[40.817386627197266]], [[44.72731399536133]], [[43.65764236450195]], [[46.42644500732422]], [[48.89083480834961]], [[45.73055648803711]], [[45.65370178222656]], [[44.44060134887695]], [[44.08531951904297]], [[44.43910598754883]], [[45.04508590698242]], [[38.68388748168945]], [[42.89270782470703]], [[43.38323211669922]], [[40.792789459228516]], [[42.09685134887695]], [[46.13134002685547]], [[45.7238883972168]], [[47.314598083496094]], [[47.06987380981445]], [[42.84346389770508]]]], dtype='float32').reshape([1, 24, 1, 1]),
        ]


class TestPrimitiveOp_fa0da2a3b7012e5372aaa197dd71ff61(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[21.57964324951172]], [[20.849185943603516]], [[21.389863967895508]], [[22.92589569091797]], [[21.12099838256836]], [[21.001768112182617]], [[21.905405044555664]], [[20.621068954467773]], [[20.97715950012207]], [[20.474952697753906]], [[21.735191345214844]], [[19.871593475341797]], [[20.53399085998535]], [[22.35291862487793]], [[20.863351821899414]], [[21.015045166015625]], [[21.126745223999023]], [[21.539566040039062]], [[19.160127639770508]], [[21.543785095214844]], [[19.366960525512695]], [[22.470134735107422]], [[19.518627166748047]], [[22.569242477416992]]]], dtype='float32').reshape([1, 24, 1, 1]),
        ]


class TestPrimitiveOp_abdf9f427a1506129d606043873b71fc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[35946.859375]], [[30698.365234375]], [[34966.9765625]], [[28342.60546875]], [[35975.96875]], [[31918.951171875]]]], dtype='float32').reshape([1, 6, 1, 1]),
        ]


class TestPrimitiveOp_e8e4c3810d9697cc7a704f770f2b52d6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[32257.3515625]], [[42317.828125]], [[36184.05078125]], [[43102.14453125]], [[41055.6953125]], [[42209.75]]]], dtype='float32').reshape([1, 6, 1, 1]),
        ]


class TestPrimitiveOp_440b256e09a4ea6286c98a5c014b3c2f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[43831.0625]], [[38675.953125]], [[44727.51953125]], [[49158.13671875]], [[41155.8046875]], [[33434.2421875]]]], dtype='float32').reshape([1, 6, 1, 1]),
        ]


class TestPrimitiveOp_8367de96c50f9d24c8694b2d95d12218(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[38319.57421875]], [[47223.0859375]], [[31635.189453125]], [[44665.33203125]], [[37245.75390625]], [[39227.7265625]]]], dtype='float32').reshape([1, 6, 1, 1]),
        ]


class TestPrimitiveOp_9b990a6bf997b08e87fc1f4a33b13c73(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 11, 17], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d43e57aa76dae90b491308a51147aeea(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.uniform([1, 240, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_51d79759447e7eae9e133d320f3b2de0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.uniform([1, 72, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3aba818f675f64a187e5d0e69893ee97(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 88, 132], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f54b39415aa1a564381e0c0ed2176c84(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[6.259962558746338]], [[6.480357646942139]], [[6.354485511779785]], [[6.02998685836792]], [[6.0928874015808105]], [[5.784942626953125]], [[5.731517314910889]], [[5.884207725524902]], [[5.937573432922363]], [[5.833317279815674]], [[5.603264808654785]], [[6.116853713989258]], [[6.1093668937683105]], [[6.023929595947266]], [[5.659120082855225]], [[5.759819507598877]], [[5.9197893142700195]], [[6.5027594566345215]], [[5.223503589630127]], [[5.036237716674805]], [[5.466402530670166]], [[6.359437942504883]], [[5.307007312774658]], [[6.603766441345215]]]], dtype='float32').reshape([1, 24, 1, 1]),
        ]


class TestPrimitiveOp_4313f504cd2df18a6be4cdf50d75c8c2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 100, 152], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c535e30a60f37f624eb57a0180b0f8e8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1c21e9be9a7e71dc79ebb27483a54b38
    def get_inputs(self):
        return [
            paddle.uniform([1, 156], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_fb8f67f9e9c1678f4f4c8b0e74def09f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.uniform([1, 168, 1, 1], dtype='float32', min=0, max=0.5),
        ]




if __name__ == '__main__':
    unittest.main()