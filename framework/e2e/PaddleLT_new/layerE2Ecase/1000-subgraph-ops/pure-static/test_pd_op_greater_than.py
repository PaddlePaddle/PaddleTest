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



class PrimitiveOp_ad7b859ae1d953909466d330af2f88e5(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 > input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 2100], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_52f00a17c08f35f70058da803bf30737(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ad7b859ae1d953909466d330af2f88e5
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


class PrimitiveOp_423e9eb54ea9d55f1bda8c5fcd9a1cb9(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 > input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[2], dtype='int32'),
            paddle.static.InputSpec(shape=[], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_8976cf2dc005978a2e6bf70c9311187c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_423e9eb54ea9d55f1bda8c5fcd9a1cb9
    def get_inputs(self):
        return [
            paddle.to_tensor([2, 6], dtype='int32').reshape([2]),
            paddle.to_tensor(-1, dtype='int32').reshape([]),
        ]


class PrimitiveOp_ef091d5ae3051289fd1a91fbf0a20c08(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 > input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[6], dtype='float32'),
            paddle.static.InputSpec(shape=[6], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_f10834ec9659192a3e29b9ed2bad20ca(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ef091d5ae3051289fd1a91fbf0a20c08
    def get_inputs(self):
        return [
            paddle.to_tensor([0.21949312090873718, 0.0012134052813053131, 0.4766252934932709, 0.11636532098054886, 0.2342921495437622, 0.4063635468482971], dtype='float32').reshape([6]),
            paddle.to_tensor([0.446471244096756, 0.4040623903274536, 0.4889226257801056, 0.043019529432058334, 0.20853398740291595, 0.4344382584095001], dtype='float32').reshape([6]),
        ]


class TestPrimitiveOp_4e7ca40c0175f87c8db6898577ca5dd3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ef091d5ae3051289fd1a91fbf0a20c08
    def get_inputs(self):
        return [
            paddle.to_tensor([0.031546175479888916, 0.1145879477262497, 0.332019567489624, 0.13446305692195892, 0.3072856068611145, 0.1730593889951706], dtype='float32').reshape([6]),
            paddle.to_tensor([0.3741452395915985, 0.2878594994544983, 0.47377514839172363, 0.20650611817836761, 0.1986602246761322, 0.43330734968185425], dtype='float32').reshape([6]),
        ]


class TestPrimitiveOp_a7f2c52c00d6c6b5a3e268b386ebce8b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_423e9eb54ea9d55f1bda8c5fcd9a1cb9
    def get_inputs(self):
        return [
            paddle.to_tensor([6, 9], dtype='int32').reshape([2]),
            paddle.to_tensor(-1, dtype='int32').reshape([]),
        ]


class PrimitiveOp_b208c3bda9bc683503f9a6c4e588a05b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 > input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 3549], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_0e540c423454dab770b485a659a008d4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b208c3bda9bc683503f9a6c4e588a05b
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


class PrimitiveOp_dc7198a93fb483ecef70d7d801046837(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 > input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 4116], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_4f9784c7150ae36a60e2e213559bdb10(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dc7198a93fb483ecef70d7d801046837
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