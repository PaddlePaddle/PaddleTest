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


class TestPrimitiveOp_017d9fe610b5cf3bd4c075c9f2c6fa49(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1c21e9be9a7e71dc79ebb27483a54b38
    def get_inputs(self):
        return [
            paddle.to_tensor([[4.033295631408691, 5.2313103675842285, 4.299407005310059, 4.828373432159424, 4.571896076202393, 5.049456596374512, 4.181540489196777, 4.226678371429443, 5.068587779998779, 4.667089939117432, 4.951745986938477, 4.2019782066345215, 5.1283111572265625, 4.435796737670898, 4.7257609367370605, 4.073487758636475, 4.219559669494629, 5.038308620452881]], dtype='float32').reshape([1, 18]),
        ]


class TestPrimitiveOp_350a8cc1d64e25270787dc064ad509ac(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1c21e9be9a7e71dc79ebb27483a54b38
    def get_inputs(self):
        return [
            paddle.to_tensor([[6.164876937866211, 7.601596832275391, 6.52034854888916, 7.175662040710449, 7.044514179229736, 5.55806303024292, 5.7198567390441895, 7.097987651824951, 6.263722896575928, 6.420352935791016, 6.3288493156433105, 6.925473213195801, 6.855074882507324, 7.4791765213012695, 5.940278053283691, 7.150235652923584, 6.6423749923706055, 7.581090450286865, 6.608335494995117, 6.835241317749023, 6.990002155303955, 6.843236923217773, 7.036097049713135]], dtype='float32').reshape([1, 23]),
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


class TestPrimitiveOp_408c9e7a8979e734114905a655b7a912(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[8.341363906860352]], [[8.345094680786133]], [[8.33063793182373]], [[8.891763687133789]], [[7.898582935333252]], [[8.031394004821777]], [[7.580002307891846]], [[8.074787139892578]], [[7.623562335968018]], [[8.525352478027344]], [[9.265737533569336]], [[7.251461982727051]], [[7.469519138336182]], [[8.449827194213867]], [[6.704586982727051]], [[7.329336643218994]], [[8.022858619689941]], [[7.848570346832275]], [[8.836841583251953]], [[8.346170425415039]], [[7.800962448120117]], [[8.04752254486084]], [[7.770833969116211]], [[7.5968756675720215]], [[7.5983500480651855]], [[7.384743690490723]], [[7.849663734436035]], [[7.752751350402832]], [[8.344961166381836]], [[8.521525382995605]]]], dtype='float32').reshape([1, 30, 1, 1]),
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


class TestPrimitiveOp_c8fdb895f799eeeed5345ecf9ffd9ab2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[8.393464088439941]], [[8.366008758544922]], [[8.483171463012695]], [[8.66169261932373]], [[8.777335166931152]], [[8.07479476928711]], [[8.880349159240723]], [[8.67391300201416]], [[8.541731834411621]], [[7.9030442237854]], [[7.630783557891846]], [[8.261388778686523]], [[8.464116096496582]], [[8.459275245666504]], [[8.572575569152832]], [[8.322819709777832]], [[9.108028411865234]], [[8.179729461669922]], [[8.376543045043945]], [[7.334713459014893]], [[8.867982864379883]], [[8.163381576538086]], [[8.024945259094238]], [[8.204413414001465]], [[8.69442081451416]], [[7.682368755340576]], [[7.935717582702637]], [[8.263702392578125]], [[7.737281322479248]], [[7.696122646331787]]]], dtype='float32').reshape([1, 30, 1, 1]),
        ]


class TestPrimitiveOp_cb6e562ed929755429e92268d85b9633(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 50, 76], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_0d3409bb3750fe90cf6f213bac2a1222(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[1.1495932340621948]], [[1.3737449645996094]], [[1.5405479669570923]], [[1.6220264434814453]], [[1.3335896730422974]]]], dtype='float32').reshape([1, 5, 1, 1]),
        ]


class TestPrimitiveOp_22b880de76fd153a6e2290ab63f95292(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[2.8691492080688477]], [[2.6245875358581543]], [[2.3653178215026855]], [[2.166766405105591]], [[2.307394027709961]], [[2.743351697921753]], [[2.8373217582702637]], [[2.3402910232543945]], [[2.6781842708587646]], [[2.5950942039489746]]]], dtype='float32').reshape([1, 10, 1, 1]),
        ]


class TestPrimitiveOp_d43e57aa76dae90b491308a51147aeea(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.uniform([1, 240, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_aedde6856ccf127f00f9f962754fed15(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[7.132702350616455]], [[6.780191898345947]], [[5.928580284118652]], [[7.082545757293701]], [[6.498932361602783]], [[6.148091793060303]], [[6.590491771697998]], [[6.749575138092041]], [[6.537431240081787]], [[6.507152557373047]], [[6.074974536895752]], [[6.766306400299072]], [[5.595406532287598]], [[6.529210090637207]], [[7.224152088165283]], [[6.767131328582764]], [[6.367806911468506]], [[6.925047397613525]], [[5.66684627532959]], [[6.8891825675964355]], [[7.272857666015625]], [[7.604793071746826]], [[7.038225173950195]], [[5.851436138153076]]]], dtype='float32').reshape([1, 24, 1, 1]),
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


class TestPrimitiveOp_438d36127d8507f890e2f848b680cdaa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[4.524796962738037]], [[5.10005521774292]], [[4.674687385559082]], [[3.8786232471466064]], [[5.400389671325684]], [[4.489811897277832]], [[5.00974702835083]], [[4.751856803894043]], [[4.238439559936523]], [[4.446267127990723]], [[4.643568515777588]], [[4.354197978973389]], [[4.672003746032715]], [[4.485158443450928]], [[5.109180450439453]], [[4.733465194702148]], [[4.423995494842529]], [[4.591418266296387]]]], dtype='float32').reshape([1, 18, 1, 1]),
        ]


class TestPrimitiveOp_d43e57aa76dae90b491308a51147aeea(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.uniform([1, 240, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a4752edc842f8d7ac1a98768a410177b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[6.493211269378662]], [[6.386541366577148]], [[7.420287609100342]], [[7.333436965942383]], [[6.9355950355529785]], [[5.896524906158447]], [[6.753997325897217]], [[6.382160663604736]], [[6.178173542022705]], [[6.534293174743652]], [[6.505472183227539]], [[6.507557392120361]], [[6.691187858581543]], [[6.243559837341309]], [[6.23176383972168]], [[6.602023124694824]], [[6.351990222930908]], [[6.180388927459717]], [[6.737544536590576]], [[6.839259147644043]], [[7.146553993225098]], [[7.609710693359375]], [[7.2124433517456055]], [[6.125067710876465]]]], dtype='float32').reshape([1, 24, 1, 1]),
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


class TestPrimitiveOp_6135d2cc495d50aa4789ee58bc97f3c2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[1.0480040311813354]], [[1.208018183708191]], [[1.4790951013565063]], [[1.330366849899292]]]], dtype='float32').reshape([1, 4, 1, 1]),
        ]


class TestPrimitiveOp_2308dc4c012a75634d23dd4b43fc5c16(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1c21e9be9a7e71dc79ebb27483a54b38
    def get_inputs(self):
        return [
            paddle.uniform([145, 84], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_24c84c6807110ab948a994762c1683dc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[3.2938592433929443]], [[3.408022165298462]], [[3.3406291007995605]], [[3.058248519897461]], [[3.145562171936035]], [[3.171884536743164]], [[3.1627354621887207]], [[3.6159627437591553]], [[2.859837055206299]], [[3.311866044998169]], [[3.52898907661438]]]], dtype='float32').reshape([1, 11, 1, 1]),
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


class TestPrimitiveOp_55154d660c010bb73dbd22204f9d792f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[7.96597957611084]], [[7.284008979797363]], [[8.071146965026855]], [[8.760475158691406]], [[8.251751899719238]], [[7.766061782836914]], [[8.217127799987793]], [[9.190387725830078]], [[7.507880210876465]], [[9.322604179382324]], [[8.097240447998047]], [[7.6459431648254395]], [[7.849544525146484]], [[8.649650573730469]], [[7.423149108886719]], [[7.311344146728516]], [[7.924724578857422]], [[9.053340911865234]], [[9.133321762084961]], [[8.5658597946167]], [[8.386237144470215]], [[7.7710771560668945]], [[8.541682243347168]], [[7.844338417053223]], [[8.349851608276367]], [[7.915876865386963]], [[8.554839134216309]], [[8.232840538024902]], [[8.90272045135498]], [[8.765819549560547]]]], dtype='float32').reshape([1, 30, 1, 1]),
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


class TestPrimitiveOp_7927adc99c5827f0607b66dde108c99e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[3.758634567260742]], [[5.082086563110352]], [[4.171976089477539]], [[3.97188663482666]], [[4.684258937835693]], [[4.020765781402588]], [[4.118278980255127]], [[4.071583271026611]], [[4.342146396636963]], [[3.982689142227173]], [[3.666200637817383]], [[4.092992305755615]], [[4.814263820648193]], [[4.547019958496094]], [[3.568892240524292]], [[3.9223012924194336]]]], dtype='float32').reshape([1, 16, 1, 1]),
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


class TestPrimitiveOp_49bd33509e67ebb48615d4f9d09d356d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[7.051290988922119]], [[8.006352424621582]], [[8.431422233581543]], [[8.261886596679688]], [[7.675865650177002]], [[7.161308765411377]], [[8.586992263793945]], [[7.49012565612793]], [[6.815402984619141]], [[7.385681629180908]], [[7.945213317871094]], [[7.4884138107299805]], [[7.443033218383789]], [[7.670413017272949]], [[7.075477123260498]], [[7.630590438842773]], [[7.0837931632995605]], [[7.8320512771606445]], [[8.103216171264648]], [[7.677859306335449]], [[7.417149543762207]], [[7.247952938079834]], [[6.757908821105957]], [[7.820258617401123]], [[8.067765235900879]], [[7.257658958435059]], [[9.007942199707031]], [[7.396412372589111]], [[8.738997459411621]], [[6.798262596130371]]]], dtype='float32').reshape([1, 30, 1, 1]),
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


class TestPrimitiveOp_e274718af586e83acb892ec9ebb6bffa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[5.892675399780273]], [[6.968152046203613]], [[6.0143609046936035]], [[6.4988789558410645]], [[5.598384380340576]], [[5.6200270652771]], [[5.922160625457764]], [[6.1982197761535645]], [[6.238831043243408]], [[6.700442314147949]], [[6.08521842956543]], [[6.50688362121582]], [[6.094415187835693]], [[6.457919597625732]], [[6.121939182281494]], [[6.495532512664795]], [[5.121353626251221]], [[6.021153450012207]], [[6.228803634643555]], [[6.077028751373291]], [[6.203137397766113]], [[5.798994064331055]], [[4.395704746246338]], [[5.763790607452393]], [[6.030169486999512]]]], dtype='float32').reshape([1, 25, 1, 1]),
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


class TestPrimitiveOp_998565c8fe5ddf9dc908fc24857b6e7a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[5.78770112991333]], [[5.173911094665527]], [[5.828648090362549]], [[5.295206069946289]], [[5.438414096832275]], [[5.914935111999512]], [[5.354312419891357]], [[5.041213512420654]], [[5.047346115112305]], [[5.584328651428223]], [[6.354373455047607]], [[5.3775105476379395]], [[5.281128406524658]], [[5.093140602111816]], [[4.823072910308838]], [[6.270191669464111]], [[5.606038570404053]], [[6.288185119628906]], [[5.714950084686279]], [[5.566715717315674]]]], dtype='float32').reshape([1, 20, 1, 1]),
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


class TestPrimitiveOp_b3340356b0e557d3f11d1d28d7f56923(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[5.502219200134277]], [[5.28275203704834]], [[5.1589226722717285]], [[4.98879861831665]], [[5.378580093383789]], [[5.263552188873291]], [[5.741299629211426]], [[5.16142463684082]], [[4.911098480224609]], [[4.91981840133667]], [[5.200071334838867]], [[5.248149871826172]], [[4.898805141448975]], [[5.904212951660156]], [[5.408774375915527]], [[5.2532830238342285]], [[5.063814640045166]], [[4.795100212097168]]]], dtype='float32').reshape([1, 18, 1, 1]),
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


class TestPrimitiveOp_51bc72ea364f1726e4df4cbaaf4d49e6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[4.264429569244385]], [[3.588606834411621]], [[3.901416063308716]], [[4.151577472686768]], [[4.0111918449401855]], [[3.7010557651519775]], [[4.194592475891113]], [[3.618128776550293]], [[4.360840797424316]], [[4.132993221282959]], [[4.101284027099609]], [[4.483236312866211]], [[4.194448947906494]], [[3.754594326019287]], [[3.4575321674346924]], [[4.242446422576904]], [[4.110925197601318]], [[3.6827917098999023]]]], dtype='float32').reshape([1, 18, 1, 1]),
        ]


class TestPrimitiveOp_cec624690c57e95be692914c3a76b07a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1c21e9be9a7e71dc79ebb27483a54b38
    def get_inputs(self):
        return [
            paddle.uniform([10, 84], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3ed91b9d3cd92ddb14e6df6afe958af9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[6.562326908111572]], [[5.669094562530518]], [[5.329018592834473]], [[5.497073650360107]], [[5.421993732452393]], [[5.694520950317383]], [[6.503623008728027]], [[5.560434341430664]], [[5.663275241851807]], [[6.299114227294922]], [[5.959976673126221]], [[6.512989044189453]], [[6.62641716003418]], [[5.377169609069824]], [[5.499907493591309]], [[5.698286533355713]], [[5.877714157104492]], [[5.456587791442871]], [[6.124011039733887]], [[6.246454238891602]], [[5.705720901489258]], [[5.565013408660889]], [[5.930296897888184]], [[5.865280628204346]]]], dtype='float32').reshape([1, 24, 1, 1]),
        ]


class TestPrimitiveOp_e97138f5dcb67670e56dccdcab817829(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 11, 17], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_034004de266647d37ea720af867ad3e0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[6.008657455444336]], [[6.322399139404297]], [[5.315286636352539]], [[6.074441909790039]], [[5.413235664367676]], [[5.292750358581543]], [[5.546027183532715]], [[5.273697376251221]], [[5.29599142074585]], [[5.233922004699707]], [[4.632728576660156]], [[4.674685478210449]], [[5.450929164886475]], [[5.20733118057251]], [[5.647550582885742]], [[5.515830039978027]], [[5.297325134277344]], [[5.323426723480225]]]], dtype='float32').reshape([1, 18, 1, 1]),
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


class TestPrimitiveOp_8f420410a031c89bb1d2e5a0044f1ec8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[5.00966215133667]], [[4.738468170166016]], [[4.393519878387451]], [[4.130898475646973]], [[4.7210307121276855]], [[5.321023941040039]], [[4.781608581542969]], [[4.954293251037598]], [[5.046009063720703]], [[4.331708908081055]], [[4.6108527183532715]], [[4.21380090713501]], [[4.622811317443848]], [[4.676739692687988]], [[5.098243713378906]], [[4.279331684112549]], [[5.087551593780518]], [[4.904784202575684]]]], dtype='float32').reshape([1, 18, 1, 1]),
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


class TestPrimitiveOp_a9e27d781ae96400b435a6cfd5bc457f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[4.570652961730957]], [[5.290533542633057]], [[5.236232757568359]], [[4.753215789794922]], [[5.1592206954956055]], [[5.3395280838012695]], [[4.476436138153076]], [[4.3913350105285645]], [[4.970709323883057]], [[4.858014106750488]], [[5.289392471313477]], [[5.0237956047058105]], [[5.544734477996826]], [[5.3807373046875]], [[4.539039611816406]], [[4.546725749969482]], [[4.030461311340332]], [[5.68776798248291]]]], dtype='float32').reshape([1, 18, 1, 1]),
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


class TestPrimitiveOp_b10ba0e7ee2054e4fcc00c40b7d93064(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[4.044375419616699]], [[3.9695029258728027]], [[4.399412631988525]], [[4.351151943206787]], [[3.939140796661377]], [[3.736872434616089]], [[4.160721778869629]], [[3.6401002407073975]], [[3.593968629837036]], [[4.237102031707764]], [[3.89855694770813]], [[3.894055128097534]], [[3.5025439262390137]], [[3.5617258548736572]], [[3.929076671600342]], [[4.469088554382324]]]], dtype='float32').reshape([1, 16, 1, 1]),
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


class TestPrimitiveOp_a85b2b8008019c7fdbc1a25d15234d5c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[4.764894485473633]], [[4.012932777404785]], [[4.432371616363525]], [[4.805740833282471]], [[4.5551652908325195]], [[4.929264545440674]], [[4.797806262969971]], [[5.429898262023926]], [[4.413793087005615]], [[4.965513706207275]], [[5.158843517303467]], [[4.463954448699951]], [[4.748885154724121]], [[4.852672576904297]], [[5.104424476623535]], [[4.424574375152588]], [[5.957918167114258]], [[4.536571502685547]]]], dtype='float32').reshape([1, 18, 1, 1]),
        ]


class TestPrimitiveOp_7aa2751357823e1e23e6762b070c5f46(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[1.5847580432891846]], [[1.0072780847549438]], [[1.3233182430267334]], [[1.5454227924346924]]]], dtype='float32').reshape([1, 4, 1, 1]),
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


class TestPrimitiveOp_a67f7492007fcf887f7d52c673658e49(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[5.271772384643555]], [[5.875980377197266]], [[5.233264923095703]], [[5.741209030151367]], [[5.142308712005615]], [[4.55288553237915]], [[5.253786087036133]], [[5.284027099609375]], [[4.941991806030273]], [[5.884860992431641]], [[5.917901992797852]], [[5.081498146057129]], [[5.755356788635254]], [[5.243422031402588]], [[5.570818901062012]], [[4.73895788192749]], [[5.321943759918213]], [[5.601650238037109]], [[5.280106067657471]], [[5.202432155609131]]]], dtype='float32').reshape([1, 20, 1, 1]),
        ]


class TestPrimitiveOp_b4b88fda1d68de0fa5cd6205016818c3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.uniform([1, 84, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f4e3d860587cb4787c3a916e8e0f8162(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[3.4511351585388184]], [[3.596243381500244]], [[3.079944610595703]], [[3.0792460441589355]], [[3.197981834411621]], [[3.155919313430786]], [[2.571645975112915]], [[3.180319309234619]], [[3.193372964859009]], [[3.010362386703491]], [[2.657683849334717]], [[3.003628969192505]]]], dtype='float32').reshape([1, 12, 1, 1]),
        ]


class TestPrimitiveOp_e69399c34d611353deb07c891beae707(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[4.712578773498535]], [[4.405738830566406]], [[5.340959548950195]], [[5.020174980163574]], [[5.04586935043335]], [[5.332529544830322]], [[5.075929164886475]], [[5.159215450286865]], [[4.925020694732666]], [[4.820794105529785]], [[4.712738513946533]], [[5.557010650634766]], [[4.693427085876465]], [[5.183993339538574]], [[4.675652980804443]], [[4.937038898468018]], [[5.127405643463135]], [[5.4516282081604]], [[5.081034183502197]], [[5.154932022094727]]]], dtype='float32').reshape([1, 20, 1, 1]),
        ]


class TestPrimitiveOp_d874e2f215b6e15daf34c07b66120f9c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[3.3770663738250732]], [[3.018279552459717]], [[3.1935417652130127]], [[3.3590152263641357]], [[3.3767566680908203]], [[3.033186197280884]], [[3.3185150623321533]], [[3.309736728668213]], [[3.105422019958496]], [[3.1335480213165283]], [[2.7805066108703613]]]], dtype='float32').reshape([1, 11, 1, 1]),
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


class TestPrimitiveOp_a2a8a0d90f9e7a70ef9c1e95acff7a6e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[3.7868828773498535]], [[4.309795379638672]], [[3.672903060913086]], [[3.462696075439453]], [[4.04031229019165]], [[3.203916072845459]], [[4.003729820251465]], [[3.761460065841675]], [[3.279228687286377]], [[3.416935443878174]], [[3.2144813537597656]], [[3.2578201293945312]], [[3.7585110664367676]], [[3.6805272102355957]]]], dtype='float32').reshape([1, 14, 1, 1]),
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


class TestPrimitiveOp_8195ff59da6d679f46555d5bf2cad3cf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[4.974609375]], [[4.976262092590332]], [[4.851878643035889]], [[4.950084209442139]], [[5.452333450317383]], [[4.530028820037842]], [[5.880236625671387]], [[5.714767932891846]], [[5.143693923950195]], [[4.772283554077148]], [[5.391656875610352]], [[5.164602756500244]], [[5.560387134552002]], [[5.347257137298584]], [[6.138763904571533]], [[5.2376790046691895]], [[5.05681037902832]], [[5.90244197845459]], [[5.363766193389893]], [[5.793654918670654]]]], dtype='float32').reshape([1, 20, 1, 1]),
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


class TestPrimitiveOp_62980106d297f5d5d79d3b33105fd57a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[30611.162109375]], [[34098.68359375]], [[33542.65234375]], [[37874.97265625]], [[35785.0390625]], [[30797.640625]]], [[[29474.904296875]], [[32840.2265625]], [[32297.748046875]], [[36470.22265625]], [[34456.5078125]], [[29652.517578125]]]], dtype='float32').reshape([2, 6, 1, 1]),
        ]


class TestPrimitiveOp_0d607570d2011f2552b94686aa20f533(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[43153.4375]], [[39524.2578125]], [[34366.5703125]], [[27617.447265625]], [[39732.4453125]], [[34258.45703125]]], [[[41476.13671875]], [[37983.9921875]], [[33026.08984375]], [[26542.349609375]], [[38184.8671875]], [[32926.66796875]]]], dtype='float32').reshape([2, 6, 1, 1]),
        ]


class TestPrimitiveOp_9b640688af901caac44922a0c9dbe648(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[46980.84375]], [[49469.58984375]], [[34778.47265625]], [[41731.0703125]], [[44243.390625]], [[45118.13671875]]], [[[44473.92578125]], [[46829.0859375]], [[32919.1796875]], [[39499.72265625]], [[41876.26171875]], [[42702.77734375]]]], dtype='float32').reshape([2, 6, 1, 1]),
        ]


class TestPrimitiveOp_9cc0f806d7be7b69b2c9fa0a8343d28b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[48133.4765625]], [[41553.1015625]], [[39663.78515625]], [[38959.29296875]], [[47156.734375]], [[43251.9921875]]], [[[45779.3359375]], [[39522.046875]], [[37720.71875]], [[37059.18359375]], [[44854.62890625]], [[41139.7734375]]]], dtype='float32').reshape([2, 6, 1, 1]),
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


class TestPrimitiveOp_0ac89b509a74cf62b155a3310eddc605(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[7.534451484680176]], [[7.903776168823242]], [[9.045938491821289]], [[7.593854904174805]], [[7.122854709625244]], [[8.03244686126709]], [[7.656867027282715]], [[7.268033027648926]], [[7.397475719451904]], [[7.093584060668945]], [[8.421623229980469]], [[8.075343132019043]], [[7.709201812744141]], [[7.9542317390441895]], [[7.758068084716797]], [[7.648263931274414]], [[6.878692150115967]], [[8.587495803833008]], [[8.04971981048584]], [[6.730032444000244]], [[7.359399318695068]], [[8.338031768798828]], [[7.228526592254639]], [[8.696207046508789]], [[7.203897476196289]], [[6.891290664672852]], [[7.43543004989624]], [[7.630880832672119]], [[7.928101062774658]], [[7.90075159072876]]]], dtype='float32').reshape([1, 30, 1, 1]),
        ]


class TestPrimitiveOp_5eeb05a6aa39371505230d5caf846db1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[7.7160964012146]], [[8.8593111038208]], [[7.149313926696777]], [[7.997802734375]], [[7.889408588409424]], [[8.159114837646484]], [[8.61439323425293]], [[8.108988761901855]], [[7.816248893737793]], [[8.484747886657715]], [[7.5950236320495605]], [[8.478062629699707]], [[7.839389324188232]], [[8.435235977172852]], [[7.654603958129883]], [[7.260209083557129]], [[7.277437210083008]], [[7.615996837615967]], [[8.556742668151855]], [[8.04092788696289]], [[7.731863498687744]], [[7.525420665740967]], [[8.144274711608887]], [[8.088244438171387]], [[8.052762985229492]], [[8.022008895874023]], [[8.075109481811523]], [[8.527801513671875]], [[7.209123134613037]], [[7.500683307647705]]]], dtype='float32').reshape([1, 30, 1, 1]),
        ]


class TestPrimitiveOp_1e094fae8e2cdd7473b4e7bc248fadfa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 44, 66], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b3951b1884df171be665492b7507bbf7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[8.071221351623535]], [[8.483712196350098]], [[7.512153625488281]], [[7.391823768615723]], [[7.566353797912598]], [[7.654898643493652]], [[8.039820671081543]], [[8.686598777770996]], [[8.012916564941406]], [[6.911598205566406]], [[8.386569023132324]], [[8.96761417388916]], [[8.04858112335205]], [[7.871091842651367]], [[7.638219833374023]], [[7.861978054046631]], [[8.267280578613281]], [[7.594769477844238]], [[9.33325481414795]], [[8.84327507019043]], [[7.729835033416748]], [[8.22775650024414]], [[8.440366744995117]], [[7.035523891448975]], [[8.408729553222656]], [[7.929727077484131]], [[7.558557033538818]], [[8.91915512084961]], [[7.66169548034668]], [[8.36217212677002]]]], dtype='float32').reshape([1, 30, 1, 1]),
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


class TestPrimitiveOp_e1430205933627390dd6b5d53fdde321(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[8.112114906311035]], [[8.157021522521973]], [[8.03063678741455]], [[7.852174758911133]], [[8.124262809753418]], [[8.385584831237793]], [[7.801120758056641]], [[7.293982028961182]], [[7.800896644592285]], [[8.04080581665039]], [[8.630279541015625]], [[7.310001373291016]], [[7.316909313201904]], [[8.372149467468262]], [[7.76572322845459]], [[8.136466979980469]], [[7.5545573234558105]], [[7.791067123413086]], [[7.560868263244629]], [[7.211294651031494]], [[7.960395812988281]], [[7.222629547119141]], [[7.690578937530518]], [[7.012873649597168]], [[8.222583770751953]], [[7.545907020568848]], [[7.361056327819824]], [[7.651693820953369]], [[8.053607940673828]], [[8.189615249633789]]]], dtype='float32').reshape([1, 30, 1, 1]),
        ]


class TestPrimitiveOp_1c2dd9a8e73eb51f1b7b11f4ee1b8772(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[3.5401933193206787]], [[3.6423428058624268]], [[3.0797853469848633]], [[3.44569730758667]], [[3.7470006942749023]], [[3.3460614681243896]], [[3.2297539710998535]], [[3.406916618347168]], [[3.404538154602051]], [[3.162586212158203]], [[3.4384658336639404]], [[2.78239369392395]]]], dtype='float32').reshape([1, 12, 1, 1]),
        ]


class TestPrimitiveOp_cc0c37421e51e9aa8ca2e0d0f89ae691(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[3.8037774562835693]], [[3.8427181243896484]], [[3.481290578842163]], [[3.867964029312134]], [[3.9030420780181885]], [[3.2974624633789062]], [[3.8915672302246094]], [[3.6613614559173584]], [[3.5692715644836426]], [[3.7305221557617188]], [[3.3878841400146484]], [[3.550487518310547]]]], dtype='float32').reshape([1, 12, 1, 1]),
        ]


class TestPrimitiveOp_62a66de176e09e986ef5d8f66b0f8aff(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[6.2766432762146]], [[6.426517009735107]], [[6.425320148468018]], [[6.268661022186279]], [[6.211418628692627]], [[6.589906692504883]], [[6.614333152770996]], [[6.040896415710449]], [[6.378408908843994]], [[5.200664520263672]], [[5.978342533111572]], [[6.291723251342773]], [[6.274717330932617]], [[6.295009136199951]], [[6.354930400848389]], [[7.007253170013428]], [[5.763204574584961]], [[6.385103702545166]], [[6.056008815765381]], [[6.913020133972168]], [[6.038397312164307]], [[6.664734840393066]], [[7.555033206939697]], [[5.833787441253662]], [[5.681970596313477]]]], dtype='float32').reshape([1, 25, 1, 1]),
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


class TestPrimitiveOp_43fb73c53f8cda02513d803951e3c2c3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[5.109279632568359]], [[5.014555931091309]], [[4.3691840171813965]], [[4.839135646820068]], [[4.961926460266113]], [[4.7770094871521]], [[4.772217273712158]], [[5.030905723571777]], [[4.831387996673584]], [[4.453762531280518]], [[4.867634296417236]], [[4.604258060455322]], [[4.665245056152344]], [[5.421208381652832]], [[4.70947265625]], [[4.673716068267822]], [[5.502685546875]], [[4.979745388031006]]]], dtype='float32').reshape([1, 18, 1, 1]),
        ]


class TestPrimitiveOp_33f4a4436633b20f880732c633def918(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1c21e9be9a7e71dc79ebb27483a54b38
    def get_inputs(self):
        return [
            paddle.uniform([1, 39], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_abb8fc68c4bc0e55016ccf7a6ec84b57(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[1.5986579656600952]], [[2.3876352310180664]], [[1.7807573080062866]], [[1.909144401550293]], [[2.1521639823913574]]]], dtype='float32').reshape([1, 5, 1, 1]),
        ]


class TestPrimitiveOp_1315d080ffe911d1d3009dece0e2ccc0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[2.8739213943481445]], [[2.9864583015441895]], [[2.4922564029693604]], [[3.1521081924438477]], [[2.8864846229553223]], [[2.758599281311035]], [[3.44002103805542]], [[3.126253843307495]], [[3.232464551925659]], [[2.122493267059326]]]], dtype='float32').reshape([1, 10, 1, 1]),
        ]


class TestPrimitiveOp_0ed78aefa30cbff660dede5c0d016879(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[5.8211588859558105]], [[5.969562530517578]], [[5.040027618408203]], [[5.545391082763672]], [[5.9392828941345215]], [[5.698614597320557]], [[6.020728588104248]], [[5.518492221832275]], [[5.775483131408691]], [[5.343318462371826]], [[6.327739715576172]], [[6.019591808319092]], [[5.764501571655273]], [[6.030677795410156]], [[4.94651460647583]], [[5.521027565002441]], [[5.575688362121582]], [[6.397119045257568]], [[5.447817802429199]], [[6.309340476989746]]]], dtype='float32').reshape([1, 20, 1, 1]),
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


class TestPrimitiveOp_b312e42c57c6871dfda3f590a1489b41(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[7.330204010009766]], [[7.411655902862549]], [[7.0330491065979]], [[6.3860554695129395]], [[6.124695777893066]], [[7.14809513092041]], [[7.321486473083496]], [[6.776966571807861]], [[6.833548545837402]], [[7.155159950256348]], [[7.691615104675293]], [[7.112806797027588]], [[7.824557304382324]], [[6.744721412658691]], [[6.807538032531738]], [[6.391137599945068]], [[6.088601112365723]], [[6.99222469329834]], [[6.901208877563477]], [[7.331323623657227]], [[7.388026237487793]], [[6.604344367980957]], [[7.211657524108887]], [[6.921568870544434]]]], dtype='float32').reshape([1, 24, 1, 1]),
        ]


class TestPrimitiveOp_801d2caad76869215f563adf18fc9f5c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1c21e9be9a7e71dc79ebb27483a54b38
    def get_inputs(self):
        return [
            paddle.uniform([22, 120], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_079fcb4170f4491bd3c9e68736ea93fe(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[2.089078426361084]], [[2.8663032054901123]], [[2.343900203704834]], [[2.262293815612793]], [[2.3554248809814453]], [[2.629438638687134]], [[2.2276692390441895]], [[2.408189058303833]], [[2.5986056327819824]], [[2.2128899097442627]]]], dtype='float32').reshape([1, 10, 1, 1]),
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


class TestPrimitiveOp_b8e8badfc981a81aac3dc53e416da46b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[4.703721523284912]], [[4.271613121032715]], [[4.065333843231201]], [[3.9800946712493896]], [[4.790715217590332]], [[4.447109222412109]], [[4.510074615478516]], [[4.803034782409668]], [[4.372105121612549]], [[4.575693130493164]], [[3.1528213024139404]], [[4.1718058586120605]], [[4.4475884437561035]], [[4.449507713317871]], [[4.6612372398376465]], [[4.065131664276123]], [[4.700448036193848]], [[4.4108357429504395]]]], dtype='float32').reshape([1, 18, 1, 1]),
        ]


class TestPrimitiveOp_9225daad975c89c97be89b23378f86a6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1c21e9be9a7e71dc79ebb27483a54b38
    def get_inputs(self):
        return [
            paddle.to_tensor([[8.505111694335938, 8.07934284210205, 7.672472953796387, 8.140864372253418, 8.518659591674805, 8.234241485595703, 8.765296936035156, 9.243855476379395, 8.8179292678833, 7.90414571762085, 9.457240104675293, 9.045259475708008, 8.633344650268555, 9.115307807922363, 8.603414535522461, 7.959333419799805, 9.288622856140137, 8.249345779418945, 8.21774959564209, 8.974778175354004, 7.656043529510498, 8.334637641906738, 8.826767921447754, 9.19527530670166, 8.036260604858398, 8.85344409942627, 7.533555030822754, 9.00255298614502, 8.619718551635742, 8.904898643493652]], dtype='float32').reshape([1, 30]),
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


class TestPrimitiveOp_ebaa4c3ac2bea7fa382a05370a600d14(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[7.3440985679626465]], [[7.084399700164795]], [[7.289212226867676]], [[6.920706748962402]], [[8.188225746154785]], [[8.090666770935059]], [[6.819272518157959]], [[7.7566986083984375]], [[7.840989112854004]], [[7.760538101196289]], [[7.646232604980469]], [[7.296329021453857]], [[6.769723415374756]], [[6.9615092277526855]], [[8.337942123413086]], [[7.496492385864258]], [[7.324259281158447]], [[7.929457187652588]], [[8.382543563842773]], [[7.682465553283691]], [[8.053009986877441]], [[6.8763203620910645]], [[6.737455368041992]], [[7.033125877380371]], [[7.677404880523682]], [[7.813915729522705]], [[7.640761375427246]], [[7.280036449432373]], [[7.833850860595703]], [[7.040121078491211]]]], dtype='float32').reshape([1, 30, 1, 1]),
        ]


class TestPrimitiveOp_5529beddaf4d193d4b3a5262c3d912c6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[1.9913126230239868]], [[1.1797109842300415]], [[1.1726734638214111]], [[1.3191273212432861]], [[1.9014294147491455]]]], dtype='float32').reshape([1, 5, 1, 1]),
        ]


class TestPrimitiveOp_0d81550f62d446aa835974b47f46d38b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[2.683842658996582]], [[2.6842782497406006]], [[2.5433034896850586]], [[2.1871120929718018]], [[2.8770945072174072]], [[2.4621968269348145]], [[2.7253870964050293]], [[2.209808349609375]], [[2.8196940422058105]], [[1.9530487060546875]]]], dtype='float32').reshape([1, 10, 1, 1]),
        ]


class TestPrimitiveOp_1c7b2ca7cbc61fce7207b57309aba9a8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[5.309731960296631]], [[6.070625305175781]], [[5.621429443359375]], [[5.663862705230713]], [[6.513097763061523]], [[5.720493316650391]], [[5.386129856109619]], [[6.173917293548584]], [[6.846288681030273]], [[6.566182613372803]], [[4.968994140625]], [[6.637065887451172]], [[5.679732322692871]], [[5.365406036376953]], [[6.678298473358154]], [[5.641069412231445]], [[5.841551303863525]], [[6.354818344116211]], [[5.365055084228516]], [[5.7638397216796875]]]], dtype='float32').reshape([1, 20, 1, 1]),
        ]


class TestPrimitiveOp_823c928d12a943ccf0179ecff50ea704(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.uniform([1, 40, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_805f80062783b9d259cca59c475aa54c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[4.353776454925537]], [[5.755500316619873]], [[4.986583709716797]], [[4.325551986694336]], [[4.293920040130615]], [[5.164376735687256]], [[4.2319016456604]], [[5.23312520980835]], [[5.250441074371338]], [[4.849178314208984]], [[4.646942138671875]], [[4.615958213806152]], [[4.809027671813965]], [[5.336738109588623]], [[4.309662342071533]], [[4.57535457611084]]]], dtype='float32').reshape([1, 16, 1, 1]),
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


class TestPrimitiveOp_c5b42dac67789dcede4c3d6685b74900(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[3.822312116622925]], [[3.8225741386413574]], [[3.9365592002868652]], [[4.422060966491699]], [[4.283381938934326]], [[4.322755813598633]], [[3.9440715312957764]], [[4.428144931793213]], [[3.9092602729797363]], [[4.2019147872924805]], [[3.6413607597351074]], [[4.337952136993408]], [[4.19197940826416]], [[4.296870708465576]]]], dtype='float32').reshape([1, 14, 1, 1]),
        ]


class TestPrimitiveOp_6e9a5ffced01453311073164bd28e69a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[5.052672386169434]], [[5.049968242645264]], [[4.773495674133301]], [[4.635282039642334]], [[4.586019992828369]], [[4.91532564163208]], [[5.271136283874512]], [[4.549180507659912]], [[5.105238914489746]], [[4.596596717834473]], [[5.254343509674072]], [[4.510046482086182]], [[4.815375328063965]], [[5.0578932762146]], [[4.837278366088867]], [[4.914398670196533]], [[4.987178325653076]], [[5.231625080108643]], [[4.814979076385498]], [[5.43070650100708]]]], dtype='float32').reshape([1, 20, 1, 1]),
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


class TestPrimitiveOp_ad572e6795d33b8e3ade9830a0f2b95f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[8.094910621643066]], [[8.318694114685059]], [[8.528326034545898]], [[8.8722562789917]], [[8.198942184448242]], [[7.374891757965088]], [[8.65483570098877]], [[7.33724308013916]], [[8.711739540100098]], [[8.610730171203613]], [[8.831395149230957]], [[7.739767551422119]], [[9.076775550842285]], [[7.7435150146484375]], [[7.959839820861816]], [[7.265451431274414]], [[8.023520469665527]], [[8.827188491821289]], [[8.824771881103516]], [[8.233180046081543]], [[8.71473217010498]], [[8.02358341217041]], [[7.9135823249816895]], [[7.4622650146484375]], [[8.667437553405762]], [[8.51819133758545]], [[7.9772047996521]], [[9.0551118850708]], [[8.486502647399902]], [[8.540457725524902]]]], dtype='float32').reshape([1, 30, 1, 1]),
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


class TestPrimitiveOp_900493513fffe339ae97c66155786080(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[6.923881530761719]], [[7.498033046722412]], [[6.0636749267578125]], [[7.236250877380371]], [[6.766568660736084]], [[5.967278957366943]], [[6.8577351570129395]], [[6.778499126434326]], [[6.799393653869629]], [[5.967048168182373]], [[7.128371715545654]], [[6.49347448348999]], [[6.275139808654785]], [[7.331699371337891]], [[6.719854831695557]], [[7.174748420715332]], [[7.164754867553711]], [[6.690661430358887]], [[5.764800548553467]], [[7.2003350257873535]], [[6.125292778015137]], [[6.216479778289795]], [[6.546446800231934]], [[6.464174270629883]]]], dtype='float32').reshape([1, 24, 1, 1]),
        ]


class TestPrimitiveOp_6736c5d71f2292c4e940478c0d09b472(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[6.346798896789551]], [[6.081398010253906]], [[7.299726963043213]], [[6.957866668701172]], [[6.698203086853027]], [[6.293685436248779]], [[7.346051216125488]], [[6.788509368896484]], [[6.625288009643555]], [[6.3033623695373535]], [[6.017172813415527]], [[6.440759658813477]], [[6.5812859535217285]], [[6.249759197235107]], [[6.78050422668457]], [[6.303239345550537]], [[5.982887268066406]], [[6.647152900695801]], [[6.8246073722839355]], [[6.154778480529785]], [[6.4065752029418945]], [[7.145350456237793]], [[6.706385612487793]], [[6.898695468902588]], [[6.413028717041016]]]], dtype='float32').reshape([1, 25, 1, 1]),
        ]


class TestPrimitiveOp_21a8b17ad311653c16b752698376b987(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[3.007822036743164]], [[2.990636110305786]], [[2.9978952407836914]], [[3.345715045928955]], [[3.1105501651763916]], [[2.689404249191284]], [[3.436768054962158]], [[2.952461004257202]], [[2.769449234008789]], [[2.8575563430786133]], [[2.4953866004943848]], [[2.7217109203338623]]]], dtype='float32').reshape([1, 12, 1, 1]),
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


class TestPrimitiveOp_cc45c9499697d1f4b688c7a5cfc1af11(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[689.8653564453125]], [[680.5704956054688]], [[664.4366455078125]], [[699.2727661132812]], [[696.021240234375]], [[744.6607055664062]], [[655.9004516601562]], [[664.6710815429688]], [[756.8004150390625]], [[715.6314697265625]], [[712.28564453125]], [[669.6095581054688]], [[680.5859375]], [[706.99365234375]], [[732.1513671875]], [[648.2433471679688]], [[719.170166015625]], [[684.26806640625]], [[769.2687377929688]], [[721.9342651367188]], [[705.001220703125]], [[705.870849609375]], [[745.766357421875]], [[657.6587524414062]]]], dtype='float32').reshape([1, 24, 1, 1]),
        ]


class TestPrimitiveOp_06a2aae08430ff2c6bb808ed34e71cf6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[77.98493194580078]], [[70.32717895507812]], [[63.509315490722656]], [[75.7135238647461]], [[63.36880111694336]], [[71.58040618896484]], [[70.64983367919922]], [[77.32463073730469]], [[65.07374572753906]], [[73.18685150146484]], [[79.83320617675781]], [[73.63390350341797]], [[78.8461685180664]], [[71.25952911376953]], [[74.20136260986328]], [[76.1542739868164]], [[66.89875030517578]], [[77.56700897216797]], [[71.27639770507812]], [[77.64783477783203]], [[76.48995971679688]], [[71.31640625]], [[75.41777801513672]], [[72.67097473144531]]]], dtype='float32').reshape([1, 24, 1, 1]),
        ]


class TestPrimitiveOp_d3f572fec8100d85c02a9f9978538115(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[34.7319450378418]], [[33.58924102783203]], [[36.77731704711914]], [[32.51150894165039]], [[32.94678497314453]], [[35.561302185058594]], [[30.16822624206543]], [[35.28459548950195]], [[35.693115234375]], [[32.90860366821289]], [[34.64813995361328]], [[34.00236129760742]], [[37.37583541870117]], [[34.15388107299805]], [[36.022220611572266]], [[35.51877212524414]], [[32.52545928955078]], [[34.947906494140625]], [[28.022186279296875]], [[37.637874603271484]], [[35.4067268371582]], [[35.718544006347656]], [[35.86259841918945]], [[34.50102996826172]]]], dtype='float32').reshape([1, 24, 1, 1]),
        ]


class TestPrimitiveOp_c064c3cc56a6e3e098840989cf342595(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[24.370454788208008]], [[22.815378189086914]], [[24.97918701171875]], [[25.29088592529297]], [[20.115402221679688]], [[24.850099563598633]], [[22.514122009277344]], [[25.646230697631836]], [[23.390384674072266]], [[25.469881057739258]], [[25.39321517944336]], [[23.23244857788086]], [[28.12644386291504]], [[20.19763946533203]], [[22.85979652404785]], [[24.788991928100586]], [[25.283416748046875]], [[26.703161239624023]], [[23.287612915039062]], [[25.041770935058594]], [[24.598798751831055]], [[25.458181381225586]], [[22.87262725830078]], [[24.76409339904785]]]], dtype='float32').reshape([1, 24, 1, 1]),
        ]


class TestPrimitiveOp_70f3bfcb19bf0d0fed30b4e0d2a79ae6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[32749.275390625]], [[39411.7109375]], [[35941.9140625]], [[26188.5]], [[36667.39453125]], [[34495.625]]]], dtype='float32').reshape([1, 6, 1, 1]),
        ]


class TestPrimitiveOp_55bfd3cc447a5b7a4fa6980af0c6e2d9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[33976.58984375]], [[38068.30078125]], [[37686.6484375]], [[31999.083984375]], [[40803.84765625]], [[39620.8984375]]]], dtype='float32').reshape([1, 6, 1, 1]),
        ]


class TestPrimitiveOp_0b70cd82e9c3044fdf548f711055f053(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[32314.748046875]], [[32420.060546875]], [[44085.0625]], [[39596.10546875]], [[40085.6328125]], [[33267.4453125]]]], dtype='float32').reshape([1, 6, 1, 1]),
        ]


class TestPrimitiveOp_fde5bd973b8240b0eee2566138305948(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[38635.25390625]], [[39958.2734375]], [[49901.5546875]], [[48418.13671875]], [[44376.90234375]], [[38225.046875]]]], dtype='float32').reshape([1, 6, 1, 1]),
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


class TestPrimitiveOp_1e19a5df3396e866d230121876974090(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[6.371476173400879]], [[6.853559970855713]], [[6.055513858795166]], [[5.596645832061768]], [[6.152677536010742]], [[6.237435340881348]], [[6.416729927062988]], [[5.967062950134277]], [[6.073554992675781]], [[6.291712284088135]], [[6.719925880432129]], [[5.367959976196289]], [[7.373897552490234]], [[5.144876003265381]], [[6.575024127960205]], [[5.4149909019470215]], [[5.995121955871582]], [[6.610832691192627]], [[5.161527633666992]], [[6.301140785217285]], [[6.229990482330322]], [[5.891112804412842]], [[5.975530624389648]], [[5.764140605926514]]]], dtype='float32').reshape([1, 24, 1, 1]),
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