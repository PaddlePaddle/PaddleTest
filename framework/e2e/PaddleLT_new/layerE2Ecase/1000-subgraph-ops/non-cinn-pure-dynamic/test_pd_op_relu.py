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


class TestPrimitiveOp_5be6b49b6e787ead85782f7a4830deb1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1c21e9be9a7e71dc79ebb27483a54b38
    def get_inputs(self):
        return [
            paddle.to_tensor([[5.0247907638549805, 4.909161567687988, 5.588427543640137, 5.066595554351807, 4.783326625823975, 4.767738342285156, 5.01627779006958, 4.4898247718811035, 4.560292720794678, 4.798861980438232, 4.785017013549805, 5.387563705444336, 4.559737682342529, 5.053497314453125, 5.301931858062744, 4.88563346862793, 4.275386333465576, 4.826330661773682]], dtype='float32').reshape([1, 18]),
        ]


class TestPrimitiveOp_0c2870fc83c3ed2dc3cd4e0759c46691(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1c21e9be9a7e71dc79ebb27483a54b38
    def get_inputs(self):
        return [
            paddle.to_tensor([[5.72562313079834, 6.078614234924316, 7.058541297912598, 6.905246257781982, 5.860133171081543, 6.706315040588379, 6.48582649230957, 6.141054153442383, 6.848719596862793, 6.397941589355469, 6.691892623901367, 6.538516998291016, 6.544531345367432, 6.585219860076904, 6.4337592124938965, 6.416224956512451, 7.118134021759033, 6.8131489753723145, 6.1631388664245605, 5.9920549392700195, 6.068944454193115, 6.223697185516357, 7.179630756378174]], dtype='float32').reshape([1, 23]),
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


class TestPrimitiveOp_67cd023d8af1f71cad4927f7c1532539(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[7.850220203399658]], [[6.410380840301514]], [[7.415620803833008]], [[7.72133731842041]], [[7.379231929779053]], [[7.465376377105713]], [[7.83229923248291]], [[7.3430914878845215]], [[8.433938980102539]], [[8.384262084960938]], [[7.256677627563477]], [[8.074450492858887]], [[8.050472259521484]], [[7.776950836181641]], [[6.175331115722656]], [[7.317542552947998]], [[8.276723861694336]], [[7.710753440856934]], [[6.962830066680908]], [[7.7543535232543945]], [[7.796797275543213]], [[8.071995735168457]], [[7.5873613357543945]], [[7.624943256378174]], [[7.499958038330078]], [[7.734288215637207]], [[8.8361234664917]], [[7.007436752319336]], [[8.493032455444336]], [[8.418644905090332]]]], dtype='float32').reshape([1, 30, 1, 1]),
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


class TestPrimitiveOp_41513f996920baf6fe65cafc35c5a42c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[6.828388690948486]], [[7.152416706085205]], [[7.548887252807617]], [[7.925854682922363]], [[8.590157508850098]], [[7.57591438293457]], [[7.858030319213867]], [[7.5591630935668945]], [[7.303663730621338]], [[7.477190971374512]], [[8.684175491333008]], [[7.87654972076416]], [[7.411413669586182]], [[7.940890312194824]], [[7.757748603820801]], [[8.416932106018066]], [[6.928280353546143]], [[7.116427421569824]], [[8.085753440856934]], [[7.5065717697143555]], [[7.985809803009033]], [[7.27017068862915]], [[7.539728164672852]], [[8.036003112792969]], [[8.134060859680176]], [[7.229857444763184]], [[7.594155788421631]], [[9.057613372802734]], [[8.551019668579102]], [[8.29045581817627]]]], dtype='float32').reshape([1, 30, 1, 1]),
        ]


class TestPrimitiveOp_cb6e562ed929755429e92268d85b9633(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 50, 76], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9a33872c44fe98542f928df61c64b76c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[1.694800853729248]], [[1.6033475399017334]], [[1.4151287078857422]], [[1.6911625862121582]], [[1.7419650554656982]]]], dtype='float32').reshape([1, 5, 1, 1]),
        ]


class TestPrimitiveOp_d976cc568b8f4a6837038a2813f70a23(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[2.871072292327881]], [[3.109473466873169]], [[3.7252604961395264]], [[2.9413692951202393]], [[2.8883273601531982]], [[2.965113401412964]], [[2.537015914916992]], [[2.602930784225464]], [[2.682055711746216]], [[3.434065818786621]]]], dtype='float32').reshape([1, 10, 1, 1]),
        ]


class TestPrimitiveOp_d43e57aa76dae90b491308a51147aeea(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.uniform([1, 240, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_506e6ccde98dd98412aae174564737cd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[6.083743572235107]], [[6.656046390533447]], [[6.4518141746521]], [[6.063625812530518]], [[5.930395603179932]], [[6.21452522277832]], [[6.806361198425293]], [[6.955392837524414]], [[6.590692043304443]], [[6.444944381713867]], [[6.871044635772705]], [[6.525993347167969]], [[6.97701358795166]], [[6.446276664733887]], [[6.60465145111084]], [[6.996643543243408]], [[6.897172451019287]], [[5.9756293296813965]], [[7.418559551239014]], [[6.6775898933410645]], [[6.010356426239014]], [[6.246400356292725]], [[6.74877405166626]], [[6.817819118499756]]]], dtype='float32').reshape([1, 24, 1, 1]),
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


class TestPrimitiveOp_3e5cc66bf7c00cb1ccf88826502bae0a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[4.866799831390381]], [[4.336925029754639]], [[4.4317803382873535]], [[5.035212993621826]], [[5.0111470222473145]], [[4.435763835906982]], [[4.339719295501709]], [[4.078125476837158]], [[4.702528953552246]], [[4.914383888244629]], [[5.69053840637207]], [[4.2484869956970215]], [[4.344150066375732]], [[4.911170482635498]], [[4.827535629272461]], [[4.324012279510498]], [[4.877218723297119]], [[4.857936382293701]]]], dtype='float32').reshape([1, 18, 1, 1]),
        ]


class TestPrimitiveOp_d43e57aa76dae90b491308a51147aeea(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.uniform([1, 240, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6adaabc8285d98e236505026cd6f9395(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[5.710595607757568]], [[5.883352756500244]], [[5.962912082672119]], [[6.112746715545654]], [[6.351295471191406]], [[5.6259589195251465]], [[5.555627346038818]], [[6.105219841003418]], [[5.789981842041016]], [[6.86287784576416]], [[5.55156135559082]], [[5.655170440673828]], [[5.961136341094971]], [[5.718842506408691]], [[5.831554412841797]], [[5.7848310470581055]], [[6.288494110107422]], [[5.579275608062744]], [[6.376399517059326]], [[6.480332374572754]], [[5.778118133544922]], [[5.190840244293213]], [[6.211253643035889]], [[5.805886745452881]]]], dtype='float32').reshape([1, 24, 1, 1]),
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


class TestPrimitiveOp_47e053a2e30c1728374eaf5a2f40d559(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[0.9315390586853027]], [[1.179887294769287]], [[1.1326204538345337]], [[1.0238231420516968]]]], dtype='float32').reshape([1, 4, 1, 1]),
        ]


class TestPrimitiveOp_2308dc4c012a75634d23dd4b43fc5c16(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1c21e9be9a7e71dc79ebb27483a54b38
    def get_inputs(self):
        return [
            paddle.uniform([145, 84], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_0c93751b6a1ac70fe3cf6f178742df81(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[3.075436592102051]], [[3.1854355335235596]], [[2.4594452381134033]], [[3.3009822368621826]], [[2.8738415241241455]], [[3.0226480960845947]], [[3.174708127975464]], [[3.2025771141052246]], [[2.724499225616455]], [[3.0178182125091553]], [[2.595012903213501]]]], dtype='float32').reshape([1, 11, 1, 1]),
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


class TestPrimitiveOp_113df218c3109b803450a6003ce15699(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[7.356210708618164]], [[7.45643949508667]], [[7.715261936187744]], [[7.937924385070801]], [[7.409104824066162]], [[8.703195571899414]], [[6.576315879821777]], [[8.27151870727539]], [[7.770561695098877]], [[7.652752876281738]], [[7.357721328735352]], [[7.527255058288574]], [[7.71347713470459]], [[6.793307304382324]], [[7.76899528503418]], [[7.068814277648926]], [[8.558565139770508]], [[7.061863422393799]], [[7.1605939865112305]], [[6.982356071472168]], [[7.739774703979492]], [[8.013056755065918]], [[7.885787487030029]], [[7.428193092346191]], [[6.9627861976623535]], [[8.001039505004883]], [[8.861226081848145]], [[7.721975326538086]], [[7.300220012664795]], [[7.781312465667725]]]], dtype='float32').reshape([1, 30, 1, 1]),
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


class TestPrimitiveOp_2a441e2a9b4d7f0ce4a70b2fb85ddcb1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[3.852346420288086]], [[3.617408275604248]], [[3.7377684116363525]], [[4.023648738861084]], [[3.9766042232513428]], [[3.867718458175659]], [[3.8445804119110107]], [[4.549490451812744]], [[4.194453239440918]], [[3.9696319103240967]], [[3.9156904220581055]], [[3.870849132537842]], [[4.273193359375]], [[4.206573009490967]], [[4.431917190551758]], [[3.9569289684295654]]]], dtype='float32').reshape([1, 16, 1, 1]),
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


class TestPrimitiveOp_329b62999172d986dda239c3f4c23ce3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[7.047674179077148]], [[7.46746301651001]], [[7.71571159362793]], [[7.386870861053467]], [[6.661334991455078]], [[7.3748087882995605]], [[7.907918930053711]], [[7.6311774253845215]], [[7.681912422180176]], [[8.161060333251953]], [[7.657646656036377]], [[7.225220203399658]], [[8.0458984375]], [[7.048724174499512]], [[7.61449670791626]], [[8.095491409301758]], [[7.270317077636719]], [[7.645841121673584]], [[7.96390438079834]], [[7.035454750061035]], [[7.8601603507995605]], [[8.187243461608887]], [[7.682562351226807]], [[6.853982448577881]], [[7.161929130554199]], [[7.546895503997803]], [[6.980509281158447]], [[7.240118980407715]], [[8.376116752624512]], [[7.675683498382568]]]], dtype='float32').reshape([1, 30, 1, 1]),
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


class TestPrimitiveOp_53703beffc242077a77958621d74065e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[7.161099433898926]], [[6.283670425415039]], [[7.077556610107422]], [[6.608377933502197]], [[6.8887434005737305]], [[7.144110202789307]], [[7.002241611480713]], [[6.787414073944092]], [[6.6316022872924805]], [[7.379030704498291]], [[7.145711421966553]], [[7.1580424308776855]], [[5.669384002685547]], [[6.9816694259643555]], [[7.313919544219971]], [[7.1442108154296875]], [[7.372506141662598]], [[6.458278179168701]], [[6.062782287597656]], [[6.468989849090576]], [[7.340620517730713]], [[6.763891696929932]], [[7.095375061035156]], [[6.321043014526367]], [[6.622008323669434]]]], dtype='float32').reshape([1, 25, 1, 1]),
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


class TestPrimitiveOp_4042e662ec21ebe4a3b0c0f634017c21(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[6.157277584075928]], [[6.654095649719238]], [[6.1927008628845215]], [[6.174805164337158]], [[5.656250953674316]], [[6.049745559692383]], [[7.229335308074951]], [[5.894186973571777]], [[5.981937885284424]], [[6.447387218475342]], [[5.8650946617126465]], [[5.305336952209473]], [[6.0240936279296875]], [[6.597114562988281]], [[6.157079696655273]], [[6.523933410644531]], [[5.694827079772949]], [[6.395604133605957]], [[6.7870025634765625]], [[5.419206142425537]]]], dtype='float32').reshape([1, 20, 1, 1]),
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


class TestPrimitiveOp_81af6c693852ec9412aa0b2b15e22002(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[5.10752534866333]], [[4.603259563446045]], [[4.440310478210449]], [[4.605095863342285]], [[4.184913158416748]], [[3.8676671981811523]], [[5.142180919647217]], [[5.176795482635498]], [[4.383977890014648]], [[3.788083076477051]], [[4.922099590301514]], [[4.851803779602051]], [[4.568479537963867]], [[4.971857070922852]], [[4.623622417449951]], [[4.299269199371338]], [[4.922940254211426]], [[3.8281266689300537]]]], dtype='float32').reshape([1, 18, 1, 1]),
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


class TestPrimitiveOp_0b934823274077806d64bd6a2335eded(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[4.549908638000488]], [[4.9383721351623535]], [[4.530771732330322]], [[4.003131866455078]], [[4.2931389808654785]], [[4.7434468269348145]], [[4.119290351867676]], [[4.080099582672119]], [[4.670665740966797]], [[4.28275728225708]], [[4.067380428314209]], [[4.824221134185791]], [[4.491727352142334]], [[5.19081974029541]], [[4.20680570602417]], [[4.18110466003418]], [[4.830199241638184]], [[4.654162883758545]]]], dtype='float32').reshape([1, 18, 1, 1]),
        ]


class TestPrimitiveOp_cec624690c57e95be692914c3a76b07a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1c21e9be9a7e71dc79ebb27483a54b38
    def get_inputs(self):
        return [
            paddle.uniform([10, 84], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_53de303d842dc85196fd3d7a3033a4e6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[6.730341911315918]], [[5.2150139808654785]], [[5.373372554779053]], [[5.788073539733887]], [[6.086791515350342]], [[6.717208385467529]], [[5.628145694732666]], [[5.626158237457275]], [[5.946242809295654]], [[5.797774791717529]], [[5.717459678649902]], [[5.4825263023376465]], [[5.418801307678223]], [[5.495123863220215]], [[5.399899959564209]], [[6.1512064933776855]], [[6.103494644165039]], [[5.643342018127441]], [[5.740949630737305]], [[5.626522064208984]], [[6.405750751495361]], [[6.131895542144775]], [[5.576231002807617]], [[6.0827131271362305]]]], dtype='float32').reshape([1, 24, 1, 1]),
        ]


class TestPrimitiveOp_e97138f5dcb67670e56dccdcab817829(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 11, 17], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d2b65da1fdd9f4057ced0c9267b4bcd4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[4.239541053771973]], [[4.816692352294922]], [[4.858314514160156]], [[5.074556350708008]], [[4.58089542388916]], [[4.850882053375244]], [[4.191903591156006]], [[4.253087520599365]], [[4.150959014892578]], [[3.8840956687927246]], [[4.621192932128906]], [[4.307068347930908]], [[5.264896392822266]], [[4.96284818649292]], [[4.513309001922607]], [[4.3990349769592285]], [[4.5447516441345215]], [[4.599179744720459]]]], dtype='float32').reshape([1, 18, 1, 1]),
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


class TestPrimitiveOp_a17aa8d59c25280c9eaeac0531c6b427(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[4.579036712646484]], [[5.627302646636963]], [[5.249024391174316]], [[6.085101127624512]], [[5.456614971160889]], [[5.158232688903809]], [[5.729126453399658]], [[5.048123359680176]], [[5.283633232116699]], [[4.933670520782471]], [[5.621869087219238]], [[4.86514139175415]], [[5.098511695861816]], [[5.634434223175049]], [[5.756320953369141]], [[5.218016147613525]], [[4.7758941650390625]], [[5.290668964385986]]]], dtype='float32').reshape([1, 18, 1, 1]),
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


class TestPrimitiveOp_568b3ea7c9cf3db65c97f405b9f93bda(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[4.774139881134033]], [[4.44288444519043]], [[4.061627388000488]], [[4.734555721282959]], [[4.272776126861572]], [[4.279536724090576]], [[4.924454689025879]], [[4.579849720001221]], [[5.035876274108887]], [[4.642727375030518]], [[5.036672592163086]], [[4.533378601074219]], [[4.076155662536621]], [[4.543557643890381]], [[4.444705486297607]], [[5.03630256652832]], [[4.620931148529053]], [[4.428459167480469]]]], dtype='float32').reshape([1, 18, 1, 1]),
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


class TestPrimitiveOp_ab1bac0f7bc6f6884cb9f9adeaff95e9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[3.887784004211426]], [[3.400664806365967]], [[3.332968235015869]], [[4.292311668395996]], [[3.8454418182373047]], [[3.7534611225128174]], [[3.7757980823516846]], [[3.8074114322662354]], [[3.3374392986297607]], [[3.346233606338501]], [[4.2629899978637695]], [[3.754770517349243]], [[3.4879889488220215]], [[3.4377477169036865]], [[2.8427469730377197]], [[3.731217861175537]]]], dtype='float32').reshape([1, 16, 1, 1]),
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


class TestPrimitiveOp_6b68c985879e12a44c31cb903e048de8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[4.966066360473633]], [[4.719254016876221]], [[5.574428558349609]], [[5.309321880340576]], [[5.40964412689209]], [[5.289345741271973]], [[5.064115047454834]], [[4.791955947875977]], [[4.5315470695495605]], [[4.723508834838867]], [[4.917330265045166]], [[5.320054531097412]], [[4.874161720275879]], [[5.4857330322265625]], [[5.171445369720459]], [[5.232664585113525]], [[4.794351577758789]], [[5.000119209289551]]]], dtype='float32').reshape([1, 18, 1, 1]),
        ]


class TestPrimitiveOp_f4c76d2ac61330b326b397f785500615(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[1.3242740631103516]], [[1.1619985103607178]], [[1.2257239818572998]], [[1.600968837738037]]]], dtype='float32').reshape([1, 4, 1, 1]),
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


class TestPrimitiveOp_a02eb78ae33204130999109ac747da72(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[5.867608547210693]], [[5.49536657333374]], [[5.5151143074035645]], [[5.808355808258057]], [[5.518977165222168]], [[5.096158504486084]], [[6.271181106567383]], [[5.590792655944824]], [[5.520811080932617]], [[5.452767848968506]], [[5.823544502258301]], [[5.6400346755981445]], [[5.126609802246094]], [[5.012604236602783]], [[5.147573471069336]], [[5.880778789520264]], [[4.466744899749756]], [[5.142038822174072]], [[5.521035194396973]], [[5.765078067779541]]]], dtype='float32').reshape([1, 20, 1, 1]),
        ]


class TestPrimitiveOp_b4b88fda1d68de0fa5cd6205016818c3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.uniform([1, 84, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_7cbba5e8225ab2fe8686e8269664051e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[3.670431613922119]], [[3.4017767906188965]], [[3.7088985443115234]], [[2.8188748359680176]], [[3.511875629425049]], [[3.398378849029541]], [[3.6883749961853027]], [[3.3284430503845215]], [[3.064164876937866]], [[3.123818874359131]], [[3.139251232147217]], [[3.20597243309021]]]], dtype='float32').reshape([1, 12, 1, 1]),
        ]


class TestPrimitiveOp_00c3972ba7223a527d78309a4f69338e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[4.653189659118652]], [[4.214825630187988]], [[5.99698543548584]], [[5.331616401672363]], [[5.504478931427002]], [[5.55187463760376]], [[5.197627544403076]], [[5.065846920013428]], [[5.104503631591797]], [[5.2176194190979]], [[5.302167892456055]], [[4.6718244552612305]], [[4.829518795013428]], [[5.492980003356934]], [[4.358083248138428]], [[5.401058197021484]], [[5.749157905578613]], [[4.652544975280762]], [[5.41537618637085]], [[5.704509258270264]]]], dtype='float32').reshape([1, 20, 1, 1]),
        ]


class TestPrimitiveOp_446be0ec3a27a5f02237ab23d8569502(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[3.541774272918701]], [[3.583669900894165]], [[3.2327818870544434]], [[3.3782896995544434]], [[2.9439098834991455]], [[3.5128774642944336]], [[3.4546477794647217]], [[3.4707159996032715]], [[2.7985730171203613]], [[3.3343186378479004]], [[2.6518492698669434]]]], dtype='float32').reshape([1, 11, 1, 1]),
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


class TestPrimitiveOp_b19e4e8273e98bf4e95c0c71ab0b62af(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[3.9662227630615234]], [[3.4809980392456055]], [[3.050659418106079]], [[3.953287124633789]], [[3.764752149581909]], [[3.1874589920043945]], [[3.9026811122894287]], [[4.052516937255859]], [[3.5598158836364746]], [[3.4012203216552734]], [[2.9586782455444336]], [[3.6440930366516113]], [[3.2894439697265625]], [[3.7816214561462402]]]], dtype='float32').reshape([1, 14, 1, 1]),
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


class TestPrimitiveOp_90755804787cad13ce64bf5c64afd969(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[5.126227378845215]], [[4.615624904632568]], [[5.39639949798584]], [[5.142935276031494]], [[5.045383453369141]], [[4.894627571105957]], [[5.579664707183838]], [[4.875969409942627]], [[5.422024726867676]], [[4.741666316986084]], [[5.0650954246521]], [[5.370575428009033]], [[4.293694496154785]], [[4.9663920402526855]], [[5.268765449523926]], [[4.810600280761719]], [[4.8917646408081055]], [[4.197942733764648]], [[5.488993167877197]], [[4.756964683532715]]]], dtype='float32').reshape([1, 20, 1, 1]),
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


class TestPrimitiveOp_74b1a92bb0aa5da4ca9fcdd2da307a75(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[42277.2890625]], [[34658.6796875]], [[37588.92578125]], [[32222.041015625]], [[29644.5390625]], [[33550.40625]]], [[[41074.41796875]], [[33686.07421875]], [[36531.6640625]], [[31303.294921875]], [[28817.443359375]], [[32597.2421875]]]], dtype='float32').reshape([2, 6, 1, 1]),
        ]


class TestPrimitiveOp_6aeeda016d9eb71849ea377e774a8d95(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[37649.17578125]], [[41821.875]], [[33838.6640625]], [[36712.5859375]], [[36409.48046875]], [[35812.41796875]]], [[[38640.72265625]], [[42923.26953125]], [[34727.796875]], [[37678.38671875]], [[37369.30078125]], [[36757.06640625]]]], dtype='float32').reshape([2, 6, 1, 1]),
        ]


class TestPrimitiveOp_7e66b07c2805d595eb2f6ae63291c3e9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[45982.49609375]], [[30480.904296875]], [[38091.53125]], [[41839.31640625]], [[38919.12890625]], [[43576.6953125]]], [[[46880.86328125]], [[31078.677734375]], [[38840.7890625]], [[42663.890625]], [[39683.58984375]], [[44432.76953125]]]], dtype='float32').reshape([2, 6, 1, 1]),
        ]


class TestPrimitiveOp_e61704e3b5ff7a0d6e166f5a3489d40c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[45499.42578125]], [[44213.3515625]], [[37121.56640625]], [[43967.328125]], [[41490.19921875]], [[45803.73828125]]], [[[45880.5234375]], [[44590.3203125]], [[37436.80078125]], [[44341.53125]], [[41841.59765625]], [[46192.51171875]]]], dtype='float32').reshape([2, 6, 1, 1]),
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


class TestPrimitiveOp_ecefec36e617188a0ef8ca5e02860f41(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[7.542322635650635]], [[7.213070869445801]], [[6.877016067504883]], [[7.558468818664551]], [[7.462998867034912]], [[7.755030632019043]], [[7.1506733894348145]], [[7.84716272354126]], [[7.4940876960754395]], [[7.207822322845459]], [[7.077805042266846]], [[7.42388916015625]], [[7.746916770935059]], [[7.274901866912842]], [[6.521085739135742]], [[7.162106990814209]], [[6.552116870880127]], [[6.899572849273682]], [[7.4951395988464355]], [[7.841178894042969]], [[8.506115913391113]], [[7.153757095336914]], [[7.740911483764648]], [[7.616919040679932]], [[7.1264472007751465]], [[7.494256973266602]], [[7.5830464363098145]], [[7.284029483795166]], [[8.21269702911377]], [[7.506575107574463]]]], dtype='float32').reshape([1, 30, 1, 1]),
        ]


class TestPrimitiveOp_aad4991887054b8c3985bbaffe875d7f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[9.350680351257324]], [[8.810539245605469]], [[8.99864387512207]], [[8.117464065551758]], [[8.394437789916992]], [[8.087057113647461]], [[8.953898429870605]], [[8.628273010253906]], [[8.878339767456055]], [[8.572881698608398]], [[8.787232398986816]], [[7.157745838165283]], [[8.177393913269043]], [[8.811300277709961]], [[9.26130485534668]], [[7.19333028793335]], [[9.088506698608398]], [[9.55791187286377]], [[8.022857666015625]], [[8.532830238342285]], [[9.058215141296387]], [[7.555534362792969]], [[8.369180679321289]], [[7.965431213378906]], [[9.163026809692383]], [[10.197159767150879]], [[9.149077415466309]], [[7.875706672668457]], [[8.75112533569336]], [[8.162129402160645]]]], dtype='float32').reshape([1, 30, 1, 1]),
        ]


class TestPrimitiveOp_1e094fae8e2cdd7473b4e7bc248fadfa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 44, 66], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_2458302db1301044f09fb4f3fa04547c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[7.295289516448975]], [[7.204415321350098]], [[7.707129001617432]], [[7.5969390869140625]], [[7.645872592926025]], [[7.568965435028076]], [[8.013806343078613]], [[7.534467697143555]], [[7.524115562438965]], [[7.817281246185303]], [[7.21710205078125]], [[6.7774739265441895]], [[7.938440799713135]], [[7.088042736053467]], [[7.350950717926025]], [[7.004399299621582]], [[7.458910942077637]], [[6.457792282104492]], [[6.906866550445557]], [[7.151056289672852]], [[7.212295055389404]], [[7.2559285163879395]], [[7.21386194229126]], [[7.163641452789307]], [[6.932377815246582]], [[7.226587295532227]], [[7.675450801849365]], [[7.0391845703125]], [[7.504758358001709]], [[6.9274210929870605]]]], dtype='float32').reshape([1, 30, 1, 1]),
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


class TestPrimitiveOp_507452fbb2253881b5779826001e31db(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[8.369719505310059]], [[7.840510845184326]], [[8.52347469329834]], [[7.7876739501953125]], [[8.8694486618042]], [[7.973025798797607]], [[8.4177885055542]], [[9.125961303710938]], [[7.797096252441406]], [[8.53489875793457]], [[7.907627105712891]], [[8.323229789733887]], [[8.71175765991211]], [[8.683496475219727]], [[8.667680740356445]], [[8.16464900970459]], [[8.238906860351562]], [[7.278522491455078]], [[9.046670913696289]], [[7.9590654373168945]], [[8.00121021270752]], [[7.9140191078186035]], [[7.0355377197265625]], [[8.831998825073242]], [[7.775221824645996]], [[8.064579010009766]], [[8.203347206115723]], [[7.545536518096924]], [[7.598377704620361]], [[9.136058807373047]]]], dtype='float32').reshape([1, 30, 1, 1]),
        ]


class TestPrimitiveOp_4b9866e36ac1267eb0a5ac714f42bbf7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[2.9090192317962646]], [[3.3313701152801514]], [[3.1470601558685303]], [[3.0634727478027344]], [[3.3448996543884277]], [[2.3325366973876953]], [[2.646023750305176]], [[2.83821964263916]], [[3.1006522178649902]], [[3.114042282104492]], [[2.988996744155884]], [[2.8920671939849854]]]], dtype='float32').reshape([1, 12, 1, 1]),
        ]


class TestPrimitiveOp_1ef7def3e836fe4a2a2065150304ae43(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[2.6413681507110596]], [[3.0041394233703613]], [[2.860701322555542]], [[2.8276419639587402]], [[3.354630947113037]], [[2.4001946449279785]], [[2.765003204345703]], [[2.947614908218384]], [[2.6335387229919434]], [[3.0719587802886963]], [[3.2401950359344482]], [[2.63741135597229]]]], dtype='float32').reshape([1, 12, 1, 1]),
        ]


class TestPrimitiveOp_f2d12eb60ac3a035f6faa0b032b9aef3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[6.43465518951416]], [[6.167778491973877]], [[6.1945672035217285]], [[5.861127853393555]], [[6.375700950622559]], [[6.514279365539551]], [[5.694643974304199]], [[6.402563095092773]], [[6.252643585205078]], [[5.755939960479736]], [[7.040493965148926]], [[6.350912094116211]], [[6.256567478179932]], [[7.086465835571289]], [[6.296385765075684]], [[7.195964336395264]], [[5.541567325592041]], [[6.9893059730529785]], [[5.531717300415039]], [[5.982404708862305]], [[6.592355728149414]], [[5.684733867645264]], [[5.915470600128174]], [[5.6908860206604]], [[6.603821754455566]]]], dtype='float32').reshape([1, 25, 1, 1]),
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


class TestPrimitiveOp_7cf15708e40ce784afee8d2239be9fa3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[4.360105991363525]], [[4.688514709472656]], [[4.720791816711426]], [[4.639763355255127]], [[4.480045795440674]], [[4.889379501342773]], [[4.068061351776123]], [[4.908852577209473]], [[4.4659833908081055]], [[5.004398822784424]], [[5.070291042327881]], [[5.3074212074279785]], [[4.330776691436768]], [[4.965434551239014]], [[3.9914205074310303]], [[4.536936283111572]], [[4.425126552581787]], [[4.817387104034424]]]], dtype='float32').reshape([1, 18, 1, 1]),
        ]


class TestPrimitiveOp_33f4a4436633b20f880732c633def918(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1c21e9be9a7e71dc79ebb27483a54b38
    def get_inputs(self):
        return [
            paddle.uniform([1, 39], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_363ce26ee316e083f34cd7ed15c98b43(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[1.690147876739502]], [[1.1477481126785278]], [[1.5876469612121582]], [[1.4030735492706299]], [[1.603986382484436]]]], dtype='float32').reshape([1, 5, 1, 1]),
        ]


class TestPrimitiveOp_fe6f0173f7d43b6968af985c6b534c41(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[2.1300649642944336]], [[3.0645670890808105]], [[2.788966178894043]], [[2.701542615890503]], [[2.479606866836548]], [[2.1792891025543213]], [[2.565462112426758]], [[2.7054295539855957]], [[2.5611631870269775]], [[2.8416919708251953]]]], dtype='float32').reshape([1, 10, 1, 1]),
        ]


class TestPrimitiveOp_03ad652f89364cb186d8fa495f7184ed(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[5.086889743804932]], [[5.554097652435303]], [[5.618411540985107]], [[5.302731037139893]], [[5.082913875579834]], [[5.409112930297852]], [[4.88266658782959]], [[5.384340286254883]], [[5.555750370025635]], [[4.858942031860352]], [[5.871918201446533]], [[5.652488708496094]], [[4.972433567047119]], [[5.516233444213867]], [[5.493321895599365]], [[4.793114185333252]], [[6.127239227294922]], [[5.900052070617676]], [[5.483379364013672]], [[5.497974395751953]]]], dtype='float32').reshape([1, 20, 1, 1]),
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


class TestPrimitiveOp_4496dee38335046dc986a29a931ad16a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[6.806848049163818]], [[6.779421806335449]], [[5.928939342498779]], [[6.151634216308594]], [[5.8947577476501465]], [[5.825343132019043]], [[6.187404155731201]], [[5.690608501434326]], [[6.611276626586914]], [[7.166670799255371]], [[7.097313404083252]], [[6.954497337341309]], [[6.246237754821777]], [[6.205399036407471]], [[5.889923572540283]], [[6.354040622711182]], [[6.265686988830566]], [[6.760129928588867]], [[6.005337238311768]], [[6.041594505310059]], [[6.07476806640625]], [[5.392578125]], [[6.428365707397461]], [[6.381635665893555]]]], dtype='float32').reshape([1, 24, 1, 1]),
        ]


class TestPrimitiveOp_801d2caad76869215f563adf18fc9f5c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1c21e9be9a7e71dc79ebb27483a54b38
    def get_inputs(self):
        return [
            paddle.uniform([22, 120], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f84bf6f777adc35902404886c2a1535f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[2.7149548530578613]], [[2.836534023284912]], [[2.2737889289855957]], [[2.4234912395477295]], [[2.16805362701416]], [[2.763005018234253]], [[3.5774857997894287]], [[1.8416321277618408]], [[2.503131866455078]], [[2.6240639686584473]]]], dtype='float32').reshape([1, 10, 1, 1]),
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


class TestPrimitiveOp_2a9d564964d8e834027a51ce074c6460(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[4.989027976989746]], [[4.333471298217773]], [[4.135684013366699]], [[5.009162902832031]], [[4.20782470703125]], [[4.118492126464844]], [[4.63630485534668]], [[4.8843817710876465]], [[4.670323848724365]], [[4.553167343139648]], [[4.147002220153809]], [[4.106956958770752]], [[4.478455066680908]], [[4.616498947143555]], [[4.093318939208984]], [[5.0294952392578125]], [[4.685277462005615]], [[5.693291187286377]]]], dtype='float32').reshape([1, 18, 1, 1]),
        ]


class TestPrimitiveOp_3ef8e7cbb7511e22919017cd11c9fe7c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1c21e9be9a7e71dc79ebb27483a54b38
    def get_inputs(self):
        return [
            paddle.to_tensor([[8.773189544677734, 8.423099517822266, 8.220671653747559, 9.318517684936523, 8.734471321105957, 8.354942321777344, 8.523990631103516, 8.453161239624023, 8.126041412353516, 8.212976455688477, 8.251940727233887, 8.193892478942871, 8.869047164916992, 8.692487716674805, 9.827540397644043, 8.897507667541504, 8.130953788757324, 8.619547843933105, 8.816047668457031, 7.91301155090332, 9.531411170959473, 7.935816287994385, 8.322325706481934, 8.152799606323242, 7.899613857269287, 9.049318313598633, 8.212468147277832, 8.3217134475708, 8.735994338989258, 8.406246185302734]], dtype='float32').reshape([1, 30]),
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


class TestPrimitiveOp_c6167250c8369ce2425afd8caff4c61c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[8.190667152404785]], [[8.532549858093262]], [[7.578472137451172]], [[8.632094383239746]], [[7.576027870178223]], [[8.524397850036621]], [[7.91270637512207]], [[7.142776012420654]], [[8.293561935424805]], [[6.992138862609863]], [[7.605541229248047]], [[7.522144794464111]], [[7.927464008331299]], [[7.343049049377441]], [[8.28248119354248]], [[7.635629653930664]], [[7.863544940948486]], [[8.674365997314453]], [[8.02641773223877]], [[6.868565559387207]], [[7.765742778778076]], [[7.794824600219727]], [[7.496888637542725]], [[7.5954389572143555]], [[7.824265956878662]], [[7.748753070831299]], [[7.126996994018555]], [[7.634503364562988]], [[6.999751091003418]], [[8.412819862365723]]]], dtype='float32').reshape([1, 30, 1, 1]),
        ]


class TestPrimitiveOp_ad7cbcfc4c734e32ab58cb2c316a96e0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[1.117078423500061]], [[1.4652667045593262]], [[1.2384942770004272]], [[1.3474791049957275]], [[1.5838005542755127]]]], dtype='float32').reshape([1, 5, 1, 1]),
        ]


class TestPrimitiveOp_f9a734115802e80a258b3f4c830f5d3f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[2.344630718231201]], [[2.803612232208252]], [[2.660125732421875]], [[2.699988842010498]], [[2.7060811519622803]], [[2.6608357429504395]], [[2.113220453262329]], [[2.41262149810791]], [[2.4208219051361084]], [[2.095384359359741]]]], dtype='float32').reshape([1, 10, 1, 1]),
        ]


class TestPrimitiveOp_ce271a9fc1adfa79a89a1d7eefc4b240(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[6.292038440704346]], [[5.7622833251953125]], [[5.8628973960876465]], [[5.231494426727295]], [[5.926563739776611]], [[5.8816022872924805]], [[4.67780065536499]], [[4.886355876922607]], [[5.886889457702637]], [[5.720672130584717]], [[5.529699802398682]], [[6.2631025314331055]], [[6.021288871765137]], [[5.579195022583008]], [[5.917940616607666]], [[4.742470741271973]], [[5.772171497344971]], [[4.694761276245117]], [[5.614266872406006]], [[5.412294864654541]]]], dtype='float32').reshape([1, 20, 1, 1]),
        ]


class TestPrimitiveOp_823c928d12a943ccf0179ecff50ea704(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.uniform([1, 40, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_44dbbc9645902cc175ec9de4cd451dd3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[3.885272979736328]], [[3.995682716369629]], [[4.380467891693115]], [[4.238519668579102]], [[3.807973861694336]], [[3.770780086517334]], [[3.4668946266174316]], [[3.885648250579834]], [[3.979804039001465]], [[3.3920586109161377]], [[4.023132801055908]], [[3.702303409576416]], [[3.5543558597564697]], [[3.9580507278442383]], [[4.54077672958374]], [[4.028207302093506]]]], dtype='float32').reshape([1, 16, 1, 1]),
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


class TestPrimitiveOp_ca11f9c8cf89fae9690232d66a608206(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[3.51597261428833]], [[3.731383800506592]], [[3.5915372371673584]], [[3.3545918464660645]], [[3.65403151512146]], [[3.4771029949188232]], [[3.5580673217773438]], [[3.1740853786468506]], [[3.2493107318878174]], [[3.4338717460632324]], [[3.2165613174438477]], [[3.9413301944732666]], [[3.2475860118865967]], [[3.414360523223877]]]], dtype='float32').reshape([1, 14, 1, 1]),
        ]


class TestPrimitiveOp_5ab63a4c7710dca8ebf17474c737931c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[5.418937683105469]], [[5.241125106811523]], [[5.984535217285156]], [[5.723231792449951]], [[5.40896463394165]], [[5.961812973022461]], [[5.400176048278809]], [[5.480480670928955]], [[6.17966365814209]], [[6.024230003356934]], [[5.300405502319336]], [[6.840229511260986]], [[5.395784378051758]], [[5.9628729820251465]], [[5.627788543701172]], [[5.5482025146484375]], [[5.286175727844238]], [[5.988452911376953]], [[5.808948516845703]], [[5.572681427001953]]]], dtype='float32').reshape([1, 20, 1, 1]),
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


class TestPrimitiveOp_0c6221dbd50929012cc4371bbafa6507(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[6.764605522155762]], [[7.043685436248779]], [[7.240815162658691]], [[7.279595851898193]], [[6.5231523513793945]], [[7.445433616638184]], [[6.549773693084717]], [[7.3310017585754395]], [[7.319559097290039]], [[7.149440765380859]], [[6.387151718139648]], [[7.872252464294434]], [[7.2891998291015625]], [[8.173873901367188]], [[7.665742874145508]], [[6.962970733642578]], [[6.7176642417907715]], [[7.236079216003418]], [[8.199427604675293]], [[7.267073154449463]], [[7.563003063201904]], [[7.959856033325195]], [[6.962826251983643]], [[7.467273235321045]], [[6.880935192108154]], [[6.395427703857422]], [[7.35816764831543]], [[7.327852249145508]], [[7.032649517059326]], [[7.583899974822998]]]], dtype='float32').reshape([1, 30, 1, 1]),
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


class TestPrimitiveOp_92648d3dbf58bcc376ef78d46d91cf47(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[6.254993915557861]], [[6.970305442810059]], [[6.205292701721191]], [[6.2595343589782715]], [[5.769108772277832]], [[5.224167346954346]], [[6.379481792449951]], [[5.747693061828613]], [[5.659907817840576]], [[5.943023204803467]], [[6.025867462158203]], [[5.703178882598877]], [[5.720294952392578]], [[6.185833930969238]], [[5.226645469665527]], [[6.315010070800781]], [[6.4829301834106445]], [[6.07181978225708]], [[5.83260440826416]], [[6.747703552246094]], [[5.934568405151367]], [[6.243119239807129]], [[6.224294185638428]], [[5.637502670288086]]]], dtype='float32').reshape([1, 24, 1, 1]),
        ]


class TestPrimitiveOp_9f160b9d879baf5ba74341586ea43152(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[6.551765441894531]], [[6.629490852355957]], [[5.867996692657471]], [[6.1684465408325195]], [[6.284104347229004]], [[6.519019603729248]], [[6.52275276184082]], [[6.112610816955566]], [[6.900456428527832]], [[6.185481071472168]], [[6.595438003540039]], [[6.794672012329102]], [[6.027585029602051]], [[6.001448631286621]], [[6.552099704742432]], [[5.967398643493652]], [[6.044076442718506]], [[6.4735331535339355]], [[6.647521495819092]], [[6.414856910705566]], [[6.089977741241455]], [[6.816417694091797]], [[5.752189636230469]], [[6.891372203826904]], [[6.464967250823975]]]], dtype='float32').reshape([1, 25, 1, 1]),
        ]


class TestPrimitiveOp_1cf23a9a1d7f96dd3b08cd984663562f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[3.279315710067749]], [[3.2554545402526855]], [[3.3334555625915527]], [[3.2763848304748535]], [[3.1578352451324463]], [[3.2209408283233643]], [[3.4248428344726562]], [[3.2314765453338623]], [[3.383251667022705]], [[3.0282557010650635]], [[3.131232738494873]], [[3.0485799312591553]]]], dtype='float32').reshape([1, 12, 1, 1]),
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


class TestPrimitiveOp_42185943023a8a1a281931b817611845(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[738.3297729492188]], [[772.1773071289062]], [[697.5306396484375]], [[757.9332275390625]], [[706.9006958007812]], [[710.8587036132812]], [[772.3379516601562]], [[776.5546264648438]], [[747.6640625]], [[698.0125732421875]], [[702.9410400390625]], [[719.0142211914062]], [[732.46142578125]], [[575.9503173828125]], [[770.2093505859375]], [[795.2775268554688]], [[707.408203125]], [[702.1323852539062]], [[775.8509521484375]], [[672.6448974609375]], [[686.5088500976562]], [[694.6845092773438]], [[732.1170043945312]], [[839.0545043945312]]]], dtype='float32').reshape([1, 24, 1, 1]),
        ]


class TestPrimitiveOp_c2b88efabd5cd51dc3c61e86839089d8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[88.27577209472656]], [[95.76002502441406]], [[87.78861999511719]], [[87.79698944091797]], [[88.5737533569336]], [[80.44439697265625]], [[89.31011962890625]], [[87.85270690917969]], [[84.488525390625]], [[89.33270263671875]], [[88.08332061767578]], [[84.07212829589844]], [[97.84424591064453]], [[85.0938720703125]], [[89.42194366455078]], [[83.89889526367188]], [[91.36042022705078]], [[87.6751937866211]], [[82.2608413696289]], [[94.42173767089844]], [[97.65007019042969]], [[84.28276824951172]], [[90.93608856201172]], [[96.48577117919922]]]], dtype='float32').reshape([1, 24, 1, 1]),
        ]


class TestPrimitiveOp_1230813bd25eac8afc4befb24307bb10(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[36.361244201660156]], [[35.59650421142578]], [[35.80752944946289]], [[36.66264343261719]], [[36.12427520751953]], [[30.409032821655273]], [[35.09974670410156]], [[34.62222671508789]], [[32.15555191040039]], [[31.91900062561035]], [[31.293312072753906]], [[36.2292366027832]], [[36.25039291381836]], [[37.37382125854492]], [[36.91184616088867]], [[35.765750885009766]], [[34.373870849609375]], [[37.94071578979492]], [[36.336631774902344]], [[34.920448303222656]], [[34.973148345947266]], [[35.75593948364258]], [[35.398006439208984]], [[35.97702407836914]]]], dtype='float32').reshape([1, 24, 1, 1]),
        ]


class TestPrimitiveOp_258a302cf0b53976e7cff6e6776bb417(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[25.351303100585938]], [[27.760879516601562]], [[31.212581634521484]], [[29.61737632751465]], [[27.91501808166504]], [[28.912965774536133]], [[28.17799949645996]], [[27.986726760864258]], [[29.23400115966797]], [[25.430774688720703]], [[28.742856979370117]], [[26.738697052001953]], [[24.9879207611084]], [[23.688814163208008]], [[28.042436599731445]], [[28.10509490966797]], [[27.81442642211914]], [[31.373493194580078]], [[29.423311233520508]], [[26.931270599365234]], [[27.819124221801758]], [[26.642812728881836]], [[27.627708435058594]], [[31.602685928344727]]]], dtype='float32').reshape([1, 24, 1, 1]),
        ]


class TestPrimitiveOp_4c4417556aff598ff2db11848f8a2802(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[35929.70703125]], [[36433.05859375]], [[36965.9453125]], [[32376.0078125]], [[33316.94921875]], [[33616.84375]]]], dtype='float32').reshape([1, 6, 1, 1]),
        ]


class TestPrimitiveOp_271bda3e0bf31f1cdadb513fdebaa4ea(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[49677.06640625]], [[43435.1484375]], [[36693.43359375]], [[43158.00390625]], [[33826.00390625]], [[46592.015625]]]], dtype='float32').reshape([1, 6, 1, 1]),
        ]


class TestPrimitiveOp_77e473b3e2b22150fe7191c4fba97d33(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[41937.74609375]], [[48162.34375]], [[47176.3203125]], [[36864.703125]], [[39939.66796875]], [[44321.77734375]]]], dtype='float32').reshape([1, 6, 1, 1]),
        ]


class TestPrimitiveOp_ea44ae52ebe35652a97f5e7543464023(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[36748.0625]], [[42465.46484375]], [[49356.62890625]], [[37389.99609375]], [[48896.0390625]], [[50524.62890625]]]], dtype='float32').reshape([1, 6, 1, 1]),
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


class TestPrimitiveOp_09e5dd6fcf25050fc41b29624438266f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1401b1664a71ff18b24a33df12c46d4a
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[5.0593037605285645]], [[5.837646961212158]], [[5.186929225921631]], [[5.490591049194336]], [[5.4441752433776855]], [[5.037900924682617]], [[5.012334823608398]], [[5.770382881164551]], [[5.6755781173706055]], [[5.36258602142334]], [[5.038281440734863]], [[5.764506816864014]], [[5.904150009155273]], [[5.765399932861328]], [[5.385991096496582]], [[5.383942127227783]], [[6.328894138336182]], [[5.334985256195068]], [[5.737000942230225]], [[5.298100471496582]], [[4.8958845138549805]], [[5.617877960205078]], [[5.112884521484375]], [[5.16455602645874]]]], dtype='float32').reshape([1, 24, 1, 1]),
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