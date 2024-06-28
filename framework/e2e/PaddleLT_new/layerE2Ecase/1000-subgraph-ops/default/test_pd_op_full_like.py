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



class PrimitiveOp_fb0071f87ba674da3993454ca5ca5551(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.full_like(input_0, input_1, paddle.int32, paddle.framework._current_expected_place())

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 2100], dtype='int32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_820dd220320d5d15c0dd210a202ab62b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fb0071f87ba674da3993454ca5ca5551
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1, 2100], dtype='int32'),
            paddle.to_tensor([20.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_3b6c0acf46af7a58d864495c011943d1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fb0071f87ba674da3993454ca5ca5551
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1, 2100], dtype='int32'),
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_3b6c0acf46af7a58d864495c011943d1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fb0071f87ba674da3993454ca5ca5551
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1, 2100], dtype='int32'),
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_aedd4b6c45bd78b1bb02f857fb73803e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.full_like(input_0, input_1, paddle.bool, paddle.framework._current_expected_place())

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='bool'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_0e7b3977cfada63fc05b4c22d2152fec(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_aedd4b6c45bd78b1bb02f857fb73803e
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 2100], dtype='int32'), 'bool'),
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_24c62281ed6882952b994949283228cb(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.full_like(input_0, input_1, paddle.float32, paddle.framework._current_expected_place())

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[768], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_3dbd4d2f592e6d6de4d363c7315eb522(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_24c62281ed6882952b994949283228cb
    def get_inputs(self):
        return [
            paddle.uniform([768], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_bbfdbfd8a157844e64305c00a91e2c97(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.full_like(input_0, input_1, paddle.int32, paddle.framework._current_expected_place())

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None], dtype='int32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_f10ea1c5c430e7ad879df8d48b018f0c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bbfdbfd8a157844e64305c00a91e2c97
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[2002], dtype='int32'),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_f10ea1c5c430e7ad879df8d48b018f0c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bbfdbfd8a157844e64305c00a91e2c97
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[2002], dtype='int32'),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_a03cfc0f90f75509762b743743fbf793(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bbfdbfd8a157844e64305c00a91e2c97
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1021], dtype='int32'),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_a03cfc0f90f75509762b743743fbf793(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bbfdbfd8a157844e64305c00a91e2c97
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1021], dtype='int32'),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_ae4317b4ab627ef9a409a6176bc40a55(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bbfdbfd8a157844e64305c00a91e2c97
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1002], dtype='int32'),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_ae4317b4ab627ef9a409a6176bc40a55(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bbfdbfd8a157844e64305c00a91e2c97
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1002], dtype='int32'),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_0928b1ee6987ccc1d4b631fb608c5bac(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.full_like(input_0, input_1, paddle.int32, paddle.framework._current_expected_place())

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 3549], dtype='int32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_9f2663e85ae7982d1fbf7cbc4c237271(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0928b1ee6987ccc1d4b631fb608c5bac
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1, 3549], dtype='int32'),
            paddle.to_tensor([80.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_5a1173c4575c2253a43f3c1a04271339(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0928b1ee6987ccc1d4b631fb608c5bac
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1, 3549], dtype='int32'),
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_5a1173c4575c2253a43f3c1a04271339(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0928b1ee6987ccc1d4b631fb608c5bac
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1, 3549], dtype='int32'),
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_ea1c5eb7777ff6e8662b1e3b3f535953(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_aedd4b6c45bd78b1bb02f857fb73803e
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 3549], dtype='int32'), 'bool'),
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_9029f934831e51297801cbe2b8105899(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.full_like(input_0, input_1, paddle.int32, paddle.framework._current_expected_place())

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 4116], dtype='int32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_08b3c0c2d5bc41784411ff735918828a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9029f934831e51297801cbe2b8105899
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1, 4116], dtype='int32'),
            paddle.to_tensor([20.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_231b65da22d565ebfd14c3341625640d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9029f934831e51297801cbe2b8105899
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1, 4116], dtype='int32'),
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_231b65da22d565ebfd14c3341625640d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9029f934831e51297801cbe2b8105899
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1, 4116], dtype='int32'),
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_f6e195b4ea07ee145e4739bbf402df22(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_aedd4b6c45bd78b1bb02f857fb73803e
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 4116], dtype='int32'), 'bool'),
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_9e05db7bb1a3de7f50ee7213c9a6ce8a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bbfdbfd8a157844e64305c00a91e2c97
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1027], dtype='int32'),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_9e05db7bb1a3de7f50ee7213c9a6ce8a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bbfdbfd8a157844e64305c00a91e2c97
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1027], dtype='int32'),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]




if __name__ == '__main__':
    unittest.main()