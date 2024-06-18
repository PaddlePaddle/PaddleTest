import os
os.environ['FLAGS_cinn_new_group_scheduler'] = '1'
os.environ['FLAGS_group_schedule_tiling_first'] = '1'
os.environ['FLAGS_prim_all'] = 'true'
os.environ['FLAGS_prim_enable_dynamic'] = '1'
os.environ['FLAGS_enable_pir_api'] = '1'
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



class PrimitiveOp_51a819a411abd1f82dd1461a369a5709(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.log(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_961ed340f224389fceae24a4b0115133(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_51a819a411abd1f82dd1461a369a5709
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_e95f4adde15fa3e753d64fe95a4d48ef(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.log(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_2d150caba14a04f342e5d68d39c9e4ed(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e95f4adde15fa3e753d64fe95a4d48ef
    def get_inputs(self):
        return [
            paddle.uniform([100, 80], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_2d150caba14a04f342e5d68d39c9e4ed(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e95f4adde15fa3e753d64fe95a4d48ef
    def get_inputs(self):
        return [
            paddle.uniform([100, 80], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_4d2ba317f32d155523b85d9e93c5263f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e95f4adde15fa3e753d64fe95a4d48ef
    def get_inputs(self):
        return [
            paddle.uniform([15200, 80], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_baef7e847a121ea37cb7b4f31f3ae338(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_51a819a411abd1f82dd1461a369a5709
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_a3e808ce544046c61289507591f7bdfa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e95f4adde15fa3e753d64fe95a4d48ef
    def get_inputs(self):
        return [
            paddle.uniform([300, 80], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_a3e808ce544046c61289507591f7bdfa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e95f4adde15fa3e753d64fe95a4d48ef
    def get_inputs(self):
        return [
            paddle.uniform([300, 80], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_91473b9e4600d3cafe200ba492c6064b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e95f4adde15fa3e753d64fe95a4d48ef
    def get_inputs(self):
        return [
            paddle.uniform([3800, 80], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_2d6e1448e6874b67856956f0488c9d91(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.log(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_d17fc519160cd49c0b512ead8b41638b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2d6e1448e6874b67856956f0488c9d91
    def get_inputs(self):
        return [
            paddle.uniform([1, 21824, 15], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_d17fc519160cd49c0b512ead8b41638b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2d6e1448e6874b67856956f0488c9d91
    def get_inputs(self):
        return [
            paddle.uniform([1, 21824, 15], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_91473b9e4600d3cafe200ba492c6064b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e95f4adde15fa3e753d64fe95a4d48ef
    def get_inputs(self):
        return [
            paddle.uniform([3800, 80], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_1fda6bb77954a1fd183cda7a84f38bf8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e95f4adde15fa3e753d64fe95a4d48ef
    def get_inputs(self):
        return [
            paddle.uniform([2204, 80], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_63c745dbb8817cd12d285c456fe412ee(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_51a819a411abd1f82dd1461a369a5709
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_4898f9685b6bb4b318607a6723896d16(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.log(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_ef734a0f6e19358f33892c33a82012d6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4898f9685b6bb4b318607a6723896d16
    def get_inputs(self):
        return [
            paddle.uniform([256], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_8a07dab40493b3eb568a014402284521(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e95f4adde15fa3e753d64fe95a4d48ef
    def get_inputs(self):
        return [
            paddle.uniform([950, 80], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_60dcc525ef94235835dce7cb54589ff9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_51a819a411abd1f82dd1461a369a5709
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_8bb357dafbf16258c8e56fe99d1a0176(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e95f4adde15fa3e753d64fe95a4d48ef
    def get_inputs(self):
        return [
            paddle.uniform([15200, 81], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_8bb357dafbf16258c8e56fe99d1a0176(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e95f4adde15fa3e753d64fe95a4d48ef
    def get_inputs(self):
        return [
            paddle.uniform([15200, 81], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_9ef71c7ce647d7910ab715c9ac846cd3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e95f4adde15fa3e753d64fe95a4d48ef
    def get_inputs(self):
        return [
            paddle.uniform([8816, 80], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_3783b3bcbe474c86624d0c24e9047968(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_51a819a411abd1f82dd1461a369a5709
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_efa08e6c0eb20c70075c2e2f8cf34a8f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_51a819a411abd1f82dd1461a369a5709
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_ce52dea009f34c374827bf1a40dd39c2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e95f4adde15fa3e753d64fe95a4d48ef
    def get_inputs(self):
        return [
            paddle.uniform([150, 80], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_036d59efcc6c40dd42e2efcd3eb0bd48(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e95f4adde15fa3e753d64fe95a4d48ef
    def get_inputs(self):
        return [
            paddle.uniform([70, 80], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_e4348f524857b8267ec75a79f212fe8e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_51a819a411abd1f82dd1461a369a5709
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_3babf4b8e7d756a8562f31280ac9ea27(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_51a819a411abd1f82dd1461a369a5709
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_1d761fb434da3dab3cd4bdd47112e0f5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_51a819a411abd1f82dd1461a369a5709
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_ef734a0f6e19358f33892c33a82012d6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4898f9685b6bb4b318607a6723896d16
    def get_inputs(self):
        return [
            paddle.uniform([256], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_038b3a609da43a587acdf088d9d8faa2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_51a819a411abd1f82dd1461a369a5709
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_406a83ddefa7d855f87833b4dd7c0457(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_51a819a411abd1f82dd1461a369a5709
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_4e52bb584d71261cc7c0b2f9d55f257b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e95f4adde15fa3e753d64fe95a4d48ef
    def get_inputs(self):
        return [
            paddle.uniform([950, 81], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_4e52bb584d71261cc7c0b2f9d55f257b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e95f4adde15fa3e753d64fe95a4d48ef
    def get_inputs(self):
        return [
            paddle.uniform([950, 81], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_ebc094c2cc4841a4cf586a059a54325c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e95f4adde15fa3e753d64fe95a4d48ef
    def get_inputs(self):
        return [
            paddle.uniform([70, 81], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_ebc094c2cc4841a4cf586a059a54325c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e95f4adde15fa3e753d64fe95a4d48ef
    def get_inputs(self):
        return [
            paddle.uniform([70, 81], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_ae5480b121bb883753ab920d58757906(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_51a819a411abd1f82dd1461a369a5709
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_721e0489ee58b7a4daa5a6f987e2ca92(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_51a819a411abd1f82dd1461a369a5709
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_2922d625f5ae581f2c891854646e65f3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_51a819a411abd1f82dd1461a369a5709
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_6bb9f59c38ecb33704effba131cb0cd7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e95f4adde15fa3e753d64fe95a4d48ef
    def get_inputs(self):
        return [
            paddle.uniform([551, 80], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_5312d8092c0ab650c53c125d8213cafb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e95f4adde15fa3e753d64fe95a4d48ef
    def get_inputs(self):
        return [
            paddle.uniform([3800, 81], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_5312d8092c0ab650c53c125d8213cafb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e95f4adde15fa3e753d64fe95a4d48ef
    def get_inputs(self):
        return [
            paddle.uniform([3800, 81], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_f47d760a565ba4949bba6fd79a83bb8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_51a819a411abd1f82dd1461a369a5709
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_257622711791b26625455a97f6270723(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e95f4adde15fa3e753d64fe95a4d48ef
    def get_inputs(self):
        return [
            paddle.uniform([247, 80], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_a4e3871ba658cac33a90bd805b541ec4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e95f4adde15fa3e753d64fe95a4d48ef
    def get_inputs(self):
        return [
            paddle.uniform([247, 81], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_a4e3871ba658cac33a90bd805b541ec4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e95f4adde15fa3e753d64fe95a4d48ef
    def get_inputs(self):
        return [
            paddle.uniform([247, 81], dtype='float32', min=-0.5, max=0.5),
        ]




if __name__ == '__main__':
    unittest.main()