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



class PrimitiveOp_851abb00a2070fc601fb680880ea59e0(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 < input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_67adc66127eb8d5e58954939983b19cc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_851abb00a2070fc601fb680880ea59e0
    def get_inputs(self):
        return [
            paddle.uniform([1024, 5], dtype='float32', min=0, max=0.5),
            paddle.to_tensor(0.11111100018024445, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_2c79d57b660a4e3e778a47a4dbf34f60(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_851abb00a2070fc601fb680880ea59e0
    def get_inputs(self):
        return [
            paddle.uniform([4096, 5], dtype='float32', min=0, max=0.5),
            paddle.to_tensor(0.11111100018024445, dtype='float32').reshape([]),
        ]


class PrimitiveOp_488dbad4033121c15e51966b4818934c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 < input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None], dtype='int64'),
            paddle.static.InputSpec(shape=[], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_fb7d6dd2409a63e3d081c093c8509699(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_488dbad4033121c15e51966b4818934c
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[3800], dtype='int64'),
            paddle.to_tensor(80, dtype='int64').reshape([]),
        ]


class TestPrimitiveOp_151baa14506b7a0941c50f0b445d0080(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_488dbad4033121c15e51966b4818934c
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[150], dtype='int64'),
            paddle.to_tensor(80, dtype='int64').reshape([]),
        ]


class TestPrimitiveOp_4ad7b897ecfc3d6453f9a3f5970b51cb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_488dbad4033121c15e51966b4818934c
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[40], dtype='int64'),
            paddle.to_tensor(80, dtype='int64').reshape([]),
        ]


class TestPrimitiveOp_e5142cd7edc759b908624e20c87f4591(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_488dbad4033121c15e51966b4818934c
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[3800], dtype='int64'),
            paddle.to_tensor(81, dtype='int64').reshape([]),
        ]


class TestPrimitiveOp_360fc68a9be7cdbb11d13bda8051fa80(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_488dbad4033121c15e51966b4818934c
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[15200], dtype='int64'),
            paddle.to_tensor(81, dtype='int64').reshape([]),
        ]


class TestPrimitiveOp_ee70025797df9c05f35957b2f4770d27(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_488dbad4033121c15e51966b4818934c
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[15200], dtype='int64'),
            paddle.to_tensor(80, dtype='int64').reshape([]),
        ]


class TestPrimitiveOp_29382690def0867f9fd1b7d86130cecc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_851abb00a2070fc601fb680880ea59e0
    def get_inputs(self):
        return [
            paddle.uniform([64, 5], dtype='float32', min=0, max=0.5),
            paddle.to_tensor(0.11111100018024445, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_c68cdbff2924d600a7961d1118749819(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_488dbad4033121c15e51966b4818934c
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[2204], dtype='int64'),
            paddle.to_tensor(80, dtype='int64').reshape([]),
        ]


class TestPrimitiveOp_adccdc7f2c73b34e0f6d5dde5a710725(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_488dbad4033121c15e51966b4818934c
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[70], dtype='int64'),
            paddle.to_tensor(81, dtype='int64').reshape([]),
        ]


class TestPrimitiveOp_8664900262adf6578ec9d425a016e2f8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_488dbad4033121c15e51966b4818934c
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[551], dtype='int64'),
            paddle.to_tensor(80, dtype='int64').reshape([]),
        ]


class TestPrimitiveOp_8ad049817094fccc0498ebef08c527cd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_488dbad4033121c15e51966b4818934c
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[247], dtype='int64'),
            paddle.to_tensor(80, dtype='int64').reshape([]),
        ]


class TestPrimitiveOp_5f0031ab867435798baa7e4a781cf18d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_488dbad4033121c15e51966b4818934c
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[950], dtype='int64'),
            paddle.to_tensor(80, dtype='int64').reshape([]),
        ]


class TestPrimitiveOp_08ef4192904fcd3268b9b0dc0068c307(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_488dbad4033121c15e51966b4818934c
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[8816], dtype='int64'),
            paddle.to_tensor(80, dtype='int64').reshape([]),
        ]


class TestPrimitiveOp_733dc76b962df853667f933d0c63e5b5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_851abb00a2070fc601fb680880ea59e0
    def get_inputs(self):
        return [
            paddle.uniform([16384, 5], dtype='float32', min=0, max=0.5),
            paddle.to_tensor(0.11111100018024445, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_485703c039182bf40e97bdd0260285a4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_851abb00a2070fc601fb680880ea59e0
    def get_inputs(self):
        return [
            paddle.uniform([256, 5], dtype='float32', min=0, max=0.5),
            paddle.to_tensor(0.11111100018024445, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_94deb46087b324a2e45fd9b056853563(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_488dbad4033121c15e51966b4818934c
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[247], dtype='int64'),
            paddle.to_tensor(81, dtype='int64').reshape([]),
        ]


class TestPrimitiveOp_fb7d6dd2409a63e3d081c093c8509699(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_488dbad4033121c15e51966b4818934c
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[3800], dtype='int64'),
            paddle.to_tensor(80, dtype='int64').reshape([]),
        ]


class TestPrimitiveOp_cae707ddf8db0e9dce362805aee028e4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_488dbad4033121c15e51966b4818934c
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[950], dtype='int64'),
            paddle.to_tensor(81, dtype='int64').reshape([]),
        ]


class TestPrimitiveOp_f8baa24d0803fe1b0fc6942f05c4f742(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_488dbad4033121c15e51966b4818934c
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[70], dtype='int64'),
            paddle.to_tensor(80, dtype='int64').reshape([]),
        ]




if __name__ == '__main__':
    unittest.main()