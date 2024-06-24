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


class TestPrimitiveOp_dc6a5e4c7340f98b8bfb818cdf5afe08(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_29b8f857b993dfc4e3131cbbd5f74e83
    def get_inputs(self):
        return [
            paddle.to_tensor([0.14741972088813782, 0.10565175116062164, 0.03872787579894066, 0.2721693217754364, 0.14351142942905426, 0.33666518330574036], dtype='float32').reshape([6]),
            paddle.to_tensor([0.2984810769557953, 0.1336534470319748, 0.11528350412845612, 0.4567588269710541, 0.24708279967308044, 0.3335161805152893], dtype='float32').reshape([6]),
        ]


class TestPrimitiveOp_4e86157c455455186231e71ac74592f6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_29b8f857b993dfc4e3131cbbd5f74e83
    def get_inputs(self):
        return [
            paddle.to_tensor([0.1421709507703781, 0.12108806520700455, 0.1709064394235611, 0.046536415815353394, 0.06752748787403107, 0.041682589799165726], dtype='float32').reshape([6]),
            paddle.to_tensor([0.2929052710533142, 0.40369516611099243, 0.2023414820432663, 0.36508259177207947, 0.267822802066803, 0.4557313323020935], dtype='float32').reshape([6]),
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