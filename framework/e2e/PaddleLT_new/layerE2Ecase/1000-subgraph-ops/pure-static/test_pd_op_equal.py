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



class PrimitiveOp_b0e99802109c69ade5fc364205af95aa(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.equal(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[86970], dtype='int32'),
            paddle.static.InputSpec(shape=[], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_01f9f9265b20043ccf3f42f2e2b354a4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b0e99802109c69ade5fc364205af95aa
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[86970], dtype='int32'),
            paddle.to_tensor(1, dtype='int32').reshape([]),
        ]


class PrimitiveOp_d409bdfcfa212ab9371880aac0d82df3(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.equal(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[242991], dtype='int32'),
            paddle.static.InputSpec(shape=[], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_114d47dd2cfe135dbf91d39df5840030(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d409bdfcfa212ab9371880aac0d82df3
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[242991], dtype='int32'),
            paddle.to_tensor(1, dtype='int32').reshape([]),
        ]


class PrimitiveOp_9afc87a96c545cf62471cede9056e562(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.equal(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[220968], dtype='int32'),
            paddle.static.InputSpec(shape=[], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_df1b7532d4ae44911d767d8c6e830626(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9afc87a96c545cf62471cede9056e562
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[220968], dtype='int32'),
            paddle.to_tensor(1, dtype='int32').reshape([]),
        ]


class PrimitiveOp_704d1f723ee4eb4f09c248253d6b5c8b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.equal(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[153450], dtype='int32'),
            paddle.static.InputSpec(shape=[], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_397903a95ea63aad9b5297782d3bf6c6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_704d1f723ee4eb4f09c248253d6b5c8b
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[153450], dtype='int32'),
            paddle.to_tensor(1, dtype='int32').reshape([]),
        ]


class PrimitiveOp_a110d44ba20cf38cb358299064f79316(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.equal(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[2002], dtype='int32'),
            paddle.static.InputSpec(shape=[], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_04857b5f79b9f16e37cb3e888272980b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a110d44ba20cf38cb358299064f79316
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[2002], dtype='int32'),
            paddle.to_tensor(0, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_5baf558f8ba78cac700fbdc69c4c2cea(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a110d44ba20cf38cb358299064f79316
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[2002], dtype='int32'),
            paddle.to_tensor(-1, dtype='int32').reshape([]),
        ]


class PrimitiveOp_8b622cd40e4c035adc5309ef4d01f7aa(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.equal(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[185691], dtype='int32'),
            paddle.static.InputSpec(shape=[], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_e8b49576b735d3ca92836033531b9a5b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8b622cd40e4c035adc5309ef4d01f7aa
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[185691], dtype='int32'),
            paddle.to_tensor(1, dtype='int32').reshape([]),
        ]


class PrimitiveOp_dca2a5bfbd88c7e462255da20118ecf7(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.equal(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1021], dtype='int32'),
            paddle.static.InputSpec(shape=[], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_bce8f2d1dca15f844e786dd4f011b773(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dca2a5bfbd88c7e462255da20118ecf7
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1021], dtype='int32'),
            paddle.to_tensor(0, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_3b71795851adf87054a5e1125a51d420(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dca2a5bfbd88c7e462255da20118ecf7
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1021], dtype='int32'),
            paddle.to_tensor(-1, dtype='int32').reshape([]),
        ]


class PrimitiveOp_9427ea84aabc0b8c4e869865bbf1c811(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.equal(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1002], dtype='int32'),
            paddle.static.InputSpec(shape=[], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_88a152d6605d31e94a52d286f4b18e72(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9427ea84aabc0b8c4e869865bbf1c811
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1002], dtype='int32'),
            paddle.to_tensor(0, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_bc921a4b0b773ccb47a328ea3bdbc37e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9427ea84aabc0b8c4e869865bbf1c811
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1002], dtype='int32'),
            paddle.to_tensor(-1, dtype='int32').reshape([]),
        ]


class PrimitiveOp_e04bc632d2c5b61edfb33b80a3e3af45(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.equal(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[], dtype='int32'),
            paddle.static.InputSpec(shape=[], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_595e481c00053831dde87b1de03326ae(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e04bc632d2c5b61edfb33b80a3e3af45
    def get_inputs(self):
        return [
            paddle.to_tensor(1025, dtype='int32').reshape([]),
            paddle.to_tensor(1025, dtype='int32').reshape([]),
        ]


class PrimitiveOp_703cae00a2dbf0988d1f8aceb87df33a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.equal(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[113061], dtype='int32'),
            paddle.static.InputSpec(shape=[], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_80771c506dfa4dc30d4a72145b687678(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_703cae00a2dbf0988d1f8aceb87df33a
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[113061], dtype='int32'),
            paddle.to_tensor(1, dtype='int32').reshape([]),
        ]


class PrimitiveOp_09410fc83ddab9f19335566bc5852d76(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.equal(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[205923], dtype='int32'),
            paddle.static.InputSpec(shape=[], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_c9eac59b43d7d49a5a928c7d7d124aa6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_09410fc83ddab9f19335566bc5852d76
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[205923], dtype='int32'),
            paddle.to_tensor(1, dtype='int32').reshape([]),
        ]


class PrimitiveOp_e09025c980750a9d2c3c3728c4e8ed7c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.equal(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[123783], dtype='int32'),
            paddle.static.InputSpec(shape=[], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_595ad39e9eae51d92d696b78af1a3942(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e09025c980750a9d2c3c3728c4e8ed7c
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[123783], dtype='int32'),
            paddle.to_tensor(1, dtype='int32').reshape([]),
        ]


class PrimitiveOp_dc9010a55109329fe087727bfe28c199(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.equal(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[171888], dtype='int32'),
            paddle.static.InputSpec(shape=[], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_6a7f2a1350d2e8bee100b218e06ca94e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dc9010a55109329fe087727bfe28c199
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[171888], dtype='int32'),
            paddle.to_tensor(1, dtype='int32').reshape([]),
        ]


class PrimitiveOp_e8678a92fa7dede9eeef7a14c22e0b1d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.equal(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[217413], dtype='int32'),
            paddle.static.InputSpec(shape=[], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_7c501bdcdddb7d722749b9c322b5ae32(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e8678a92fa7dede9eeef7a14c22e0b1d
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[217413], dtype='int32'),
            paddle.to_tensor(1, dtype='int32').reshape([]),
        ]


class PrimitiveOp_4c98c32a2d7a1a894dbe803a384d582a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.equal(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1027], dtype='int32'),
            paddle.static.InputSpec(shape=[], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_391e1951bf5fb6a04b64527b0106f66b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4c98c32a2d7a1a894dbe803a384d582a
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1027], dtype='int32'),
            paddle.to_tensor(0, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_ebe36331c4c4f5fd948becb3c81c71ee(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4c98c32a2d7a1a894dbe803a384d582a
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1027], dtype='int32'),
            paddle.to_tensor(-1, dtype='int32').reshape([]),
        ]


class PrimitiveOp_ddf6cfa1fdf499da89a5bd425b5b831c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.equal(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[185658], dtype='int32'),
            paddle.static.InputSpec(shape=[], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_d2c480858418a0697cc82d60fc9d0325(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ddf6cfa1fdf499da89a5bd425b5b831c
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[185658], dtype='int32'),
            paddle.to_tensor(1, dtype='int32').reshape([]),
        ]




if __name__ == '__main__':
    unittest.main()