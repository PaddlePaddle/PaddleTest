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


class PrimitiveOp_1960db1ba51b85e0d5b2e12864816327(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 > input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 500, 128], dtype='int32'),
            paddle.static.InputSpec(shape=[], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_a6394a285b0053b94aeb7eebdc79dae7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1960db1ba51b85e0d5b2e12864816327
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1, 500, 128], dtype='int32'),
            paddle.to_tensor(0, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_a6394a285b0053b94aeb7eebdc79dae7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1960db1ba51b85e0d5b2e12864816327
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


class TestPrimitiveOp_33942247bdc94c4bcea558e5e3793e67(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_29b8f857b993dfc4e3131cbbd5f74e83
    def get_inputs(self):
        return [
            paddle.to_tensor([0.056108973920345306, 0.36671507358551025, 0.13837383687496185, 0.014441891573369503, 0.30787521600723267, 0.30013731122016907], dtype='float32').reshape([6]),
            paddle.to_tensor([0.3997354805469513, 0.39295482635498047, 0.11267633736133575, 0.49924978613853455, 0.30787521600723267, 0.35715749859809875], dtype='float32').reshape([6]),
        ]


class TestPrimitiveOp_a6f0e50e9a6c8c0d98c8ac555883c3de(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_29b8f857b993dfc4e3131cbbd5f74e83
    def get_inputs(self):
        return [
            paddle.to_tensor([0.18877169489860535, 0.04359738901257515, 0.18587720394134521, 0.2778036594390869, 0.22354553639888763, 0.17368389666080475], dtype='float32').reshape([6]),
            paddle.to_tensor([0.4620789587497711, 0.33816057443618774, 0.29246312379837036, 0.4799637198448181, 0.3316073417663574, 0.26939353346824646], dtype='float32').reshape([6]),
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


class PrimitiveOp_0ccbf4ffa818ff73e88e246b6e82556b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 > input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1], dtype='int32'),
            paddle.static.InputSpec(shape=[], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_a70b1c51c96cfbccdbd660f1d4e021aa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0ccbf4ffa818ff73e88e246b6e82556b
    def get_inputs(self):
        return [
            paddle.to_tensor([7], dtype='int32').reshape([1]),
            paddle.to_tensor(0, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_3129a7ab58a86e418606adea2ca1f786(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0ccbf4ffa818ff73e88e246b6e82556b
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


class TestPrimitiveOp_a6394a285b0053b94aeb7eebdc79dae7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1960db1ba51b85e0d5b2e12864816327
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1, 500, 128], dtype='int32'),
            paddle.to_tensor(0, dtype='int32').reshape([]),
        ]




if __name__ == '__main__':
    unittest.main()