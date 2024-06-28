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



class PrimitiveOp_26458e7a6fb4cc42c5d670191054ceb3(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 > input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_f1aa02417260ca910e2f8680c53aa6be(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_26458e7a6fb4cc42c5d670191054ceb3
    def get_inputs(self):
        return [
            paddle.uniform([1, 2100], dtype='float32', min=0, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class PrimitiveOp_ff940aacfb053208f2319af432970979(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 > input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None], dtype='int32'),
            paddle.static.InputSpec(shape=[], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_2d36fa2de18539d1b7997d463904d854(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ff940aacfb053208f2319af432970979
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1, 500, 128], dtype='int32'),
            paddle.to_tensor(0, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_2d36fa2de18539d1b7997d463904d854(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ff940aacfb053208f2319af432970979
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1, 500, 128], dtype='int32'),
            paddle.to_tensor(0, dtype='int32').reshape([]),
        ]


class PrimitiveOp_589dc419d93c692b87f080aa9e3f390b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 > input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None], dtype='int32'),
            paddle.static.InputSpec(shape=[], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_78737dcce15e4129ade444595acc8507(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_589dc419d93c692b87f080aa9e3f390b
    def get_inputs(self):
        return [
            paddle.to_tensor([2, 6], dtype='int32').reshape([2]),
            paddle.to_tensor(-1, dtype='int32').reshape([]),
        ]


class PrimitiveOp_29b8f857b993dfc4e3131cbbd5f74e83(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 > input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None], dtype='float32'),
            paddle.static.InputSpec(shape=[None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_2920c3a342e28689c908a02d8b365cdd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_29b8f857b993dfc4e3131cbbd5f74e83
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0922485888004303, 0.3836251497268677, 0.3167136311531067, 0.26296818256378174, 0.4444928765296936, 0.3183137774467468], dtype='float32').reshape([6]),
            paddle.to_tensor([0.4335317313671112, 0.3333803117275238, 0.34194254875183105, 0.26296818256378174, 0.3854544162750244, 0.4297538995742798], dtype='float32').reshape([6]),
        ]


class TestPrimitiveOp_8f2cb3abd2f55d41528e16d6543375e1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_29b8f857b993dfc4e3131cbbd5f74e83
    def get_inputs(self):
        return [
            paddle.to_tensor([0.1620912402868271, 0.20841006934642792, 0.06403285264968872, 0.25364255905151367, 0.15589286386966705, 0.3629100024700165], dtype='float32').reshape([6]),
            paddle.to_tensor([0.42629650235176086, 0.3437585234642029, 0.4915377199649811, 0.19970282912254333, 0.4174017608165741, 0.3629100024700165], dtype='float32').reshape([6]),
        ]


class TestPrimitiveOp_38afe5168b6b99287b6e1412828be7a4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_589dc419d93c692b87f080aa9e3f390b
    def get_inputs(self):
        return [
            paddle.to_tensor([6, 9], dtype='int32').reshape([2]),
            paddle.to_tensor(-1, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_88a864668016b75eb79ee904d7f1ff5a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_26458e7a6fb4cc42c5d670191054ceb3
    def get_inputs(self):
        return [
            paddle.uniform([1, 3549], dtype='float32', min=0, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_0b9b459b40e62fd750aea906d2e74bb4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_589dc419d93c692b87f080aa9e3f390b
    def get_inputs(self):
        return [
            paddle.to_tensor([7], dtype='int32').reshape([1]),
            paddle.to_tensor(0, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_fa113c1ab8e86e8ffc755eeecf6aa58a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_589dc419d93c692b87f080aa9e3f390b
    def get_inputs(self):
        return [
            paddle.to_tensor([3], dtype='int32').reshape([1]),
            paddle.to_tensor(0, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_97688dc49f0ab4d302d12e9033d27468(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_26458e7a6fb4cc42c5d670191054ceb3
    def get_inputs(self):
        return [
            paddle.uniform([1, 4116], dtype='float32', min=0, max=0.5),
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_2d36fa2de18539d1b7997d463904d854(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ff940aacfb053208f2319af432970979
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1, 500, 128], dtype='int32'),
            paddle.to_tensor(0, dtype='int32').reshape([]),
        ]




if __name__ == '__main__':
    unittest.main()