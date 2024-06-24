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



class PrimitiveOp_720ad26af3b2f0ee434c1ffd071cd7a2(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = [0, 1]
        return paddle.unsqueeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_fdb506fe12314ad5ff30276a80d481d4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_720ad26af3b2f0ee434c1ffd071cd7a2
    def get_inputs(self):
        return [
            paddle.uniform([24, 36], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_fdb506fe12314ad5ff30276a80d481d4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_720ad26af3b2f0ee434c1ffd071cd7a2
    def get_inputs(self):
        return [
            paddle.uniform([24, 36], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0, 1], dtype='int64').reshape([2]),
        ]


class PrimitiveOp_1a7601697bb1f5580fda8c480301e31d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = [2, 3]
        return paddle.unsqueeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_0cf1bc7d29bd8afef81392311b86336e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1a7601697bb1f5580fda8c480301e31d
    def get_inputs(self):
        return [
            paddle.uniform([1, 72], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_a6eb330dd19fdcaac08594e115bd989d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1a7601697bb1f5580fda8c480301e31d
    def get_inputs(self):
        return [
            paddle.uniform([1, 92], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_269565e204192035c81c80ec12234cbb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1a7601697bb1f5580fda8c480301e31d
    def get_inputs(self):
        return [
            paddle.uniform([1, 960], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_db61b233c254ae808e802eab85074961(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1a7601697bb1f5580fda8c480301e31d
    def get_inputs(self):
        return [
            paddle.uniform([1, 480], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_0fa8c2c7f1716cc30316dbed50a2004b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1a7601697bb1f5580fda8c480301e31d
    def get_inputs(self):
        return [
            paddle.uniform([10, 336], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_0b0749c5f836f1c3ecab7c1c22c9feb7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1a7601697bb1f5580fda8c480301e31d
    def get_inputs(self):
        return [
            paddle.uniform([4, 80], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


class PrimitiveOp_11421ecd2ecbbe423944eb7eb8f42226(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = [1, 2]
        return paddle.unsqueeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None], dtype='float32'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_a0af31e1056396ad6c539843513f8688(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_11421ecd2ecbbe423944eb7eb8f42226
    def get_inputs(self):
        return [
            paddle.to_tensor([0.38565558195114136, 0.15376394987106323, 0.29357004165649414, 0.4372651278972626], dtype='float32').reshape([4]),
            paddle.to_tensor([1, 2], dtype='int64').reshape([2]),
        ]


class PrimitiveOp_62ceace5eda7bd790997aa9cf6734f7e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = [2]
        return paddle.unsqueeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_a1cfde6a062a2794fed86a04eeb00c26(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_62ceace5eda7bd790997aa9cf6734f7e
    def get_inputs(self):
        return [
            paddle.uniform([10, 2, 9, 112, 112], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_37546cb3989d7ca73efa74a730b190b1(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = [-1]
        return paddle.unsqueeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_ccd9b0755e05bf09d95318bff4ba0003(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_37546cb3989d7ca73efa74a730b190b1
    def get_inputs(self):
        return [
            paddle.uniform([1, 2100], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_5fbae16c52ec57d0c7462e48b41785a6(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = [2]
        return paddle.unsqueeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_56d0f7a4639efedccd7f72c1eaa2c43b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5fbae16c52ec57d0c7462e48b41785a6
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1, 500], dtype='int64'),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_f3dacf69dd92f09b122a64ffb0d6262b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = [2]
        return paddle.unsqueeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='int32'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_15485c1dfc6d7f868409cf0620c52376(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f3dacf69dd92f09b122a64ffb0d6262b
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1, 500], dtype='int32'),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_0886e70572843ebd0858ba6475167ef6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1a7601697bb1f5580fda8c480301e31d
    def get_inputs(self):
        return [
            paddle.uniform([10, 60], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


class PrimitiveOp_7df75c742412e3266ee6985318445ce4(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = [-1]
        return paddle.unsqueeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='bool'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_b744a851e03ea44fec540113ea70cea5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7df75c742412e3266ee6985318445ce4
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 8732], dtype='int32'), 'bool'),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_3f93fd19861afc5e0a23f4d1dfc43a1d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1a7601697bb1f5580fda8c480301e31d
    def get_inputs(self):
        return [
            paddle.uniform([145, 336], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


class PrimitiveOp_cb1524500f57c783066b117e2c76f2c4(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = [0]
        return paddle.unsqueeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_268440c2a538e6344619f5d3c8863fa8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cb1524500f57c783066b117e2c76f2c4
    def get_inputs(self):
        return [
            paddle.uniform([21, 256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_56d0f7a4639efedccd7f72c1eaa2c43b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5fbae16c52ec57d0c7462e48b41785a6
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1, 500], dtype='int64'),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_15485c1dfc6d7f868409cf0620c52376(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f3dacf69dd92f09b122a64ffb0d6262b
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1, 500], dtype='int32'),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_3f93fd19861afc5e0a23f4d1dfc43a1d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1a7601697bb1f5580fda8c480301e31d
    def get_inputs(self):
        return [
            paddle.uniform([145, 336], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


class PrimitiveOp_fde19d27b6aaa8e9ba51736c65cffa48(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = [0]
        return paddle.unsqueeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_bbe6ab8e28e79d37be99010ad439ba9a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fde19d27b6aaa8e9ba51736c65cffa48
    def get_inputs(self):
        return [
            paddle.uniform([1, 21824, 2], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_480ad6e1ce1a74410de5665b6bf82300(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = [2]
        return paddle.unsqueeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_5c86541f5b88af6610dbdd9928253c57(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_480ad6e1ce1a74410de5665b6bf82300
    def get_inputs(self):
        return [
            paddle.to_tensor([[[0.3121618628501892, 0.14701275527477264], [0.4195224940776825, 0.38692712783813477], [0.38966456055641174, 0.0181307103484869], [0.49889588356018066, 0.46952900290489197], [0.49599751830101013, 0.3641703426837921], [0.42518582940101624, 0.1771775335073471]]], dtype='float32').reshape([1, 6, 2]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_e169319e47e93900bde86c675f2969e0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_62ceace5eda7bd790997aa9cf6734f7e
    def get_inputs(self):
        return [
            paddle.uniform([10, 4, 49, 56, 56], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_729e4377ce6c4d2c5ec9cfcf417fbffc(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = [-1]
        return paddle.unsqueeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_7e1b3c2562f2dee49edb4352f9779ca6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_729e4377ce6c4d2c5ec9cfcf417fbffc
    def get_inputs(self):
        return [
            paddle.to_tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype='int64').reshape([16]),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_5d675aecdaea3e7e1d12836a610d6f01(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_729e4377ce6c4d2c5ec9cfcf417fbffc
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype='int64').reshape([16]),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_b5bd897d0ef26c7b549e14222b99fc79(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1a7601697bb1f5580fda8c480301e31d
    def get_inputs(self):
        return [
            paddle.uniform([145, 240], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


class PrimitiveOp_d71c8ab7c677eb4a295ddaef4ec00c78(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = [-1]
        return paddle.unsqueeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_a44cbc1e7a0f7d98a6400c20c8f8be6f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d71c8ab7c677eb4a295ddaef4ec00c78
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 19], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_5734e81a1f97314fcc2a6f70da79276d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cb1524500f57c783066b117e2c76f2c4
    def get_inputs(self):
        return [
            paddle.uniform([300, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_30dbd1828ec84343dd4014aa784dd049(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = [-2]
        return paddle.unsqueeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_ea7e653dc7dd0edeb41e5efdff5d79a7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_30dbd1828ec84343dd4014aa784dd049
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.10348998755216599, 0.044277485460042953, 0.21761225163936615, 0.07678458839654922]], dtype='float32').reshape([1, 4]),
            paddle.to_tensor([-2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_8c035f7699293ef300877cb7b7eface2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cb1524500f57c783066b117e2c76f2c4
    def get_inputs(self):
        return [
            paddle.uniform([300, 256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_c9886c1b2ee6b6b0892424aab86cff0c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = [1]
        return paddle.unsqueeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_170cc41580040ff823eb83349d84c730(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c9886c1b2ee6b6b0892424aab86cff0c
    def get_inputs(self):
        return [
            paddle.uniform([1, 36, 64, 128], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_170cc41580040ff823eb83349d84c730(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c9886c1b2ee6b6b0892424aab86cff0c
    def get_inputs(self):
        return [
            paddle.uniform([1, 36, 64, 128], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_ca81c4c15055f527e9b86605c42512bb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fde19d27b6aaa8e9ba51736c65cffa48
    def get_inputs(self):
        return [
            paddle.uniform([512, 64, 128], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_b20106555f1bc85e07ce1fd37239645d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = [2]
        return paddle.unsqueeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_2ecb8248f2878497ed210c1f0592d412(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b20106555f1bc85e07ce1fd37239645d
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 64, 128], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_c94bfd8efbfa1d73fb0c8e7b149dc95e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_720ad26af3b2f0ee434c1ffd071cd7a2
    def get_inputs(self):
        return [
            paddle.uniform([25, 38], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_c94bfd8efbfa1d73fb0c8e7b149dc95e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_720ad26af3b2f0ee434c1ffd071cd7a2
    def get_inputs(self):
        return [
            paddle.uniform([25, 38], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_c5947c308c99556cd0aca7cb8efd9d27(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d71c8ab7c677eb4a295ddaef4ec00c78
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 21], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_46ae8db885b9d7d88cb6575c35dfc48f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1a7601697bb1f5580fda8c480301e31d
    def get_inputs(self):
        return [
            paddle.uniform([3, 80], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_cb2558ef78216cc1f9edd837c33eed5d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_11421ecd2ecbbe423944eb7eb8f42226
    def get_inputs(self):
        return [
            paddle.to_tensor([0.4750056862831116, 0.3209798336029053, 0.2568384110927582], dtype='float32').reshape([3]),
            paddle.to_tensor([1, 2], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_1b449c8ceb6bf1424635675479db5c7a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1a7601697bb1f5580fda8c480301e31d
    def get_inputs(self):
        return [
            paddle.uniform([22, 60], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_621a2cbc345df2edbdb8d7cae3294180(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1a7601697bb1f5580fda8c480301e31d
    def get_inputs(self):
        return [
            paddle.uniform([1, 872], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


class PrimitiveOp_e1101708cb01271c757da4a74320dff3(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = [-1]
        return paddle.unsqueeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='int32'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_00df5e957bd1ad45883e435992d4ad45(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e1101708cb01271c757da4a74320dff3
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1, 3549], dtype='int32'),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_2b5fb95cdadf8dd4d13f7dbab493720a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = [-1]
        return paddle.unsqueeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None], dtype='float32'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_d6f3dc73cddd8803fabef4035d74a9be(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2b5fb95cdadf8dd4d13f7dbab493720a
    def get_inputs(self):
        return [
            paddle.uniform([1745], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_6e905a7d77b3bdc3caa1fd44db14b0bc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7df75c742412e3266ee6985318445ce4
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 3549], dtype='int32'), 'bool'),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_84df27c152d68c4a892169d2025e6df8(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = [-1]
        return paddle.unsqueeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_19795bfe1bdda86b087154a5e4b7dcc3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_84df27c152d68c4a892169d2025e6df8
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1745, 4], dtype='int64'),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_19795bfe1bdda86b087154a5e4b7dcc3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_84df27c152d68c4a892169d2025e6df8
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1745, 4], dtype='int64'),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_dcaae68ce2e7cfa522434b89e7ca9d7c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_720ad26af3b2f0ee434c1ffd071cd7a2
    def get_inputs(self):
        return [
            paddle.uniform([20, 30], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_dcaae68ce2e7cfa522434b89e7ca9d7c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_720ad26af3b2f0ee434c1ffd071cd7a2
    def get_inputs(self):
        return [
            paddle.uniform([20, 30], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_937184b6b0c692529686d1c9c9f0113e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_62ceace5eda7bd790997aa9cf6734f7e
    def get_inputs(self):
        return [
            paddle.uniform([22, 4, 49, 56, 56], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_b013ff8bb2829030407f69b23fdf1c6b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1a7601697bb1f5580fda8c480301e31d
    def get_inputs(self):
        return [
            paddle.uniform([171, 336], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_7addfebdcc2f5e81ff28e40c57dc7341(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_480ad6e1ce1a74410de5665b6bf82300
    def get_inputs(self):
        return [
            paddle.uniform([43, 768, 49], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_64227e5340f96fdb90370eb6b5d6aaa5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1a7601697bb1f5580fda8c480301e31d
    def get_inputs(self):
        return [
            paddle.uniform([10, 240], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_7e7d0592e4a4a6210a8ec690dbab5f43(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e1101708cb01271c757da4a74320dff3
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1, 11109], dtype='int32'),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_3ddaebd12cee7ecc188bb86fa035c367(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2b5fb95cdadf8dd4d13f7dbab493720a
    def get_inputs(self):
        return [
            paddle.uniform([5556], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_ec1bf5ae4ab26845cbbfb7334b399730(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7df75c742412e3266ee6985318445ce4
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 11109], dtype='int32'), 'bool'),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_be2ca3c7d020446f583202a3045c5e2b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_84df27c152d68c4a892169d2025e6df8
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[5556, 4], dtype='int64'),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_be2ca3c7d020446f583202a3045c5e2b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_84df27c152d68c4a892169d2025e6df8
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[5556, 4], dtype='int64'),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_ae2795a0a7d5a68276e6060f354d0da3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_729e4377ce6c4d2c5ec9cfcf417fbffc
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[36], dtype='int64'),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_ae2795a0a7d5a68276e6060f354d0da3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_729e4377ce6c4d2c5ec9cfcf417fbffc
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[36], dtype='int64'),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_0fa8c2c7f1716cc30316dbed50a2004b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1a7601697bb1f5580fda8c480301e31d
    def get_inputs(self):
        return [
            paddle.uniform([10, 336], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_b8fd754ef34ee88055b2da5213f97870(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_62ceace5eda7bd790997aa9cf6734f7e
    def get_inputs(self):
        return [
            paddle.uniform([10, 32, 49, 7, 7], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_c1f0faccb109e44e4d86d750497345f5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cb1524500f57c783066b117e2c76f2c4
    def get_inputs(self):
        return [
            paddle.uniform([19, 256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_9d625c71ae1c8efc55376e75f42c167c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1a7601697bb1f5580fda8c480301e31d
    def get_inputs(self):
        return [
            paddle.uniform([10, 36], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_f2d49023e589abc86348bf85b32bb582(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1a7601697bb1f5580fda8c480301e31d
    def get_inputs(self):
        return [
            paddle.uniform([10, 480], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_00df5e957bd1ad45883e435992d4ad45(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e1101708cb01271c757da4a74320dff3
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1, 3549], dtype='int32'),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_86017e1bdfd07a75ec2f4c3501da21f3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2b5fb95cdadf8dd4d13f7dbab493720a
    def get_inputs(self):
        return [
            paddle.uniform([1744], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_6e905a7d77b3bdc3caa1fd44db14b0bc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7df75c742412e3266ee6985318445ce4
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 3549], dtype='int32'), 'bool'),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_472c03e19fa6ba9cf1c96a6d8411c593(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_84df27c152d68c4a892169d2025e6df8
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1744, 4], dtype='int64'),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_472c03e19fa6ba9cf1c96a6d8411c593(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_84df27c152d68c4a892169d2025e6df8
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1744, 4], dtype='int64'),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_c308c6c09548ac7c6121a475e7459257(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1a7601697bb1f5580fda8c480301e31d
    def get_inputs(self):
        return [
            paddle.uniform([22, 336], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_7a4db0df2d9c314afe7af4afbb3b29e3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1a7601697bb1f5580fda8c480301e31d
    def get_inputs(self):
        return [
            paddle.uniform([171, 240], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_b013ff8bb2829030407f69b23fdf1c6b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1a7601697bb1f5580fda8c480301e31d
    def get_inputs(self):
        return [
            paddle.uniform([171, 336], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


class PrimitiveOp_9e5e04e807f5b387ca352f26f41cd915(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = [0]
        return paddle.unsqueeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None], dtype='float32'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_4535c56e0dc70e74d570936e0419a280(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9e5e04e807f5b387ca352f26f41cd915
    def get_inputs(self):
        return [
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_c6bb833eb793e68682bd802ab1d7fde6(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = [2]
        return paddle.unsqueeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_fc793a20eb093c1a42f953072c71c73d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6bb833eb793e68682bd802ab1d7fde6
    def get_inputs(self):
        return [
            paddle.uniform([1, 512], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_4ce1048cfba3a17cbacbcd63aedacd93(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = [3]
        return paddle.unsqueeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_0f5f1dddc856089eb3ca0a791494a866(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4ce1048cfba3a17cbacbcd63aedacd93
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_95372c499155b1d5c970c5a4fcc8f1ed(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_62ceace5eda7bd790997aa9cf6734f7e
    def get_inputs(self):
        return [
            paddle.uniform([22, 8, 49, 28, 28], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_c24d2073b78c9230e50a9903635d41a6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_729e4377ce6c4d2c5ec9cfcf417fbffc
    def get_inputs(self):
        return [
            paddle.to_tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype='int64').reshape([24]),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_e75cca00b600ebeea83f570cf0c894ab(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_729e4377ce6c4d2c5ec9cfcf417fbffc
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype='int64').reshape([24]),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_46b312265894b6c230dc314c140e7920(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1a7601697bb1f5580fda8c480301e31d
    def get_inputs(self):
        return [
            paddle.uniform([171, 60], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_9ea52c903563c894a6ee8c86ab2aecdc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e1101708cb01271c757da4a74320dff3
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1, 3024], dtype='int32'),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_5f37d18809c3bc58b63315626eee3f85(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2b5fb95cdadf8dd4d13f7dbab493720a
    def get_inputs(self):
        return [
            paddle.uniform([1547], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_77c0b62d6b6128ecaa1b7056cfb90eb6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7df75c742412e3266ee6985318445ce4
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 3024], dtype='int32'), 'bool'),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_1b331b17fbc0437c6b16f993961b2291(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_84df27c152d68c4a892169d2025e6df8
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1547, 4], dtype='int64'),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_1b331b17fbc0437c6b16f993961b2291(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_84df27c152d68c4a892169d2025e6df8
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1547, 4], dtype='int64'),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_8b408a88bab485affc0db8e83d1efaa0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_37546cb3989d7ca73efa74a730b190b1
    def get_inputs(self):
        return [
            paddle.uniform([1, 3549], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_9e260f8cc904738ff59297be836a1ad7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_62ceace5eda7bd790997aa9cf6734f7e
    def get_inputs(self):
        return [
            paddle.uniform([10, 16, 49, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_797b28763c5073b16d55838896ea583f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1a7601697bb1f5580fda8c480301e31d
    def get_inputs(self):
        return [
            paddle.uniform([22, 240], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_e0f295538c01e1ce4f0054d98745afa0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_729e4377ce6c4d2c5ec9cfcf417fbffc
    def get_inputs(self):
        return [
            paddle.to_tensor([0, 0, 0, 0], dtype='int64').reshape([4]),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_ab32313052232e8cacb262d7c1efbe68(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_729e4377ce6c4d2c5ec9cfcf417fbffc
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 1, 1, 1], dtype='int64').reshape([4]),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_ca81c4c15055f527e9b86605c42512bb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fde19d27b6aaa8e9ba51736c65cffa48
    def get_inputs(self):
        return [
            paddle.uniform([512, 64, 128], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_2ecb8248f2878497ed210c1f0592d412(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b20106555f1bc85e07ce1fd37239645d
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 64, 128], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_46746dcac8f078a2ba0ad9d5e8a7698e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b20106555f1bc85e07ce1fd37239645d
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 97, 97], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_266b9d7a9a6ca188ab3ddd343e2d9aac(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1a7601697bb1f5580fda8c480301e31d
    def get_inputs(self):
        return [
            paddle.uniform([22, 36], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_9e260f8cc904738ff59297be836a1ad7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_62ceace5eda7bd790997aa9cf6734f7e
    def get_inputs(self):
        return [
            paddle.uniform([10, 16, 49, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_ca81c4c15055f527e9b86605c42512bb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fde19d27b6aaa8e9ba51736c65cffa48
    def get_inputs(self):
        return [
            paddle.uniform([512, 64, 128], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_2ecb8248f2878497ed210c1f0592d412(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b20106555f1bc85e07ce1fd37239645d
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 64, 128], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_63bd453d8bf09cde95cde524202aa18a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1a7601697bb1f5580fda8c480301e31d
    def get_inputs(self):
        return [
            paddle.uniform([145, 60], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_ee694c41b0772fb74a31df068d12a654(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c9886c1b2ee6b6b0892424aab86cff0c
    def get_inputs(self):
        return [
            paddle.uniform([1, 72, 32, 64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_ee694c41b0772fb74a31df068d12a654(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c9886c1b2ee6b6b0892424aab86cff0c
    def get_inputs(self):
        return [
            paddle.uniform([1, 72, 32, 64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_00f223a733aaae7978de850a92156613(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1a7601697bb1f5580fda8c480301e31d
    def get_inputs(self):
        return [
            paddle.uniform([1, 672], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_5bb86cc78b6ac65fcb66b3f98f65d76c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_37546cb3989d7ca73efa74a730b190b1
    def get_inputs(self):
        return [
            paddle.uniform([1, 4116], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_c308c6c09548ac7c6121a475e7459257(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1a7601697bb1f5580fda8c480301e31d
    def get_inputs(self):
        return [
            paddle.uniform([22, 336], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_95372c499155b1d5c970c5a4fcc8f1ed(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_62ceace5eda7bd790997aa9cf6734f7e
    def get_inputs(self):
        return [
            paddle.uniform([22, 8, 49, 28, 28], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_1b2ef89f69602e34fe88118741570561(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e1101708cb01271c757da4a74320dff3
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1, 4116], dtype='int32'),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_7146c81008d84692f93d0a85f879bae1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2b5fb95cdadf8dd4d13f7dbab493720a
    def get_inputs(self):
        return [
            paddle.uniform([2056], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_59ab69dfccef5aceec525e33e06ecee4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7df75c742412e3266ee6985318445ce4
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 4116], dtype='int32'), 'bool'),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_b476ba66a15fa59d41868f0290c76893(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_84df27c152d68c4a892169d2025e6df8
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[2056, 4], dtype='int64'),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_b476ba66a15fa59d41868f0290c76893(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_84df27c152d68c4a892169d2025e6df8
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[2056, 4], dtype='int64'),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_d293a0a3777947d694fde3c1ba9cfe01(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e1101708cb01271c757da4a74320dff3
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1, 9261], dtype='int32'),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_d6a7fdc36e0d2c831342d59be18cca1c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2b5fb95cdadf8dd4d13f7dbab493720a
    def get_inputs(self):
        return [
            paddle.uniform([4650], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_540c939c499f4f20ac5d9667e9c6111d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7df75c742412e3266ee6985318445ce4
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 9261], dtype='int32'), 'bool'),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_4e84fbb5c1fbe3d705eaa35c997627ba(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_84df27c152d68c4a892169d2025e6df8
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[4650, 4], dtype='int64'),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_4e84fbb5c1fbe3d705eaa35c997627ba(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_84df27c152d68c4a892169d2025e6df8
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[4650, 4], dtype='int64'),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_4640200d8aadf0d92d983fe8cfe687b3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1a7601697bb1f5580fda8c480301e31d
    def get_inputs(self):
        return [
            paddle.uniform([6, 80], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_4f0beb848f5522daf583cd1319156853(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_11421ecd2ecbbe423944eb7eb8f42226
    def get_inputs(self):
        return [
            paddle.to_tensor([0.49028000235557556, 0.40219151973724365, 0.0392024964094162, 0.1044401228427887, 0.0016230230685323477, 0.3070937693119049], dtype='float32').reshape([6]),
            paddle.to_tensor([1, 2], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_a8e979b9a47f72af78b1725a7451f7ea(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7df75c742412e3266ee6985318445ce4
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 2434], dtype='int32'), 'bool'),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_db7bbac956a5ab02458c6ec34916db88(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e1101708cb01271c757da4a74320dff3
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1, 2100], dtype='int32'),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_6506969002e7d0b3717e1fc613a58c68(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2b5fb95cdadf8dd4d13f7dbab493720a
    def get_inputs(self):
        return [
            paddle.uniform([1059], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_10b84fb46bdd24602b229306bee208be(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7df75c742412e3266ee6985318445ce4
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 2100], dtype='int32'), 'bool'),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_a47b624cac5279b521f7c9af3bab1c27(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_84df27c152d68c4a892169d2025e6df8
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1059, 4], dtype='int64'),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_a47b624cac5279b521f7c9af3bab1c27(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_84df27c152d68c4a892169d2025e6df8
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1059, 4], dtype='int64'),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_754afcb23299c90b80ac4669ffce60ef(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9e5e04e807f5b387ca352f26f41cd915
    def get_inputs(self):
        return [
            paddle.to_tensor([0.4188864231109619, 0.4517028331756592, 0.15065787732601166, 0.030897535383701324], dtype='float32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_378c195022a36e0258b5cf92059ffd2c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = [1]
        return paddle.unsqueeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_07579a28594261c6555d9cd289aafd8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_378c195022a36e0258b5cf92059ffd2c
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.4188864231109619, 0.4517028331756592, 0.15065787732601166, 0.030897535383701324]], dtype='float32').reshape([1, 4]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_20b15829cace8857d79958ee7cdc612d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_30dbd1828ec84343dd4014aa784dd049
    def get_inputs(self):
        return [
            paddle.uniform([100, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_adf63c037c3b90c5ddb646e7e0cd9750(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_480ad6e1ce1a74410de5665b6bf82300
    def get_inputs(self):
        return [
            paddle.uniform([11, 768, 49], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_f4e913ac460fa620d14a06317294f371(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_62ceace5eda7bd790997aa9cf6734f7e
    def get_inputs(self):
        return [
            paddle.uniform([22, 32, 49, 7, 7], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_ca81c4c15055f527e9b86605c42512bb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fde19d27b6aaa8e9ba51736c65cffa48
    def get_inputs(self):
        return [
            paddle.uniform([512, 64, 128], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_2ecb8248f2878497ed210c1f0592d412(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b20106555f1bc85e07ce1fd37239645d
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 64, 128], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_c91202235fc0d5e126d257dcdfa2be57(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9e5e04e807f5b387ca352f26f41cd915
    def get_inputs(self):
        return [
            paddle.to_tensor([0.4614632725715637, 0.3346860110759735, 0.0953991562128067, 0.0004730565124191344], dtype='float32').reshape([4]),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_151341cc77065e735a0443907e1c2f4b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_378c195022a36e0258b5cf92059ffd2c
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.4614632725715637, 0.3346860110759735, 0.0953991562128067, 0.0004730565124191344]], dtype='float32').reshape([1, 4]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_7c7be3c0ffc801ad49f3a7a12905add0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_30dbd1828ec84343dd4014aa784dd049
    def get_inputs(self):
        return [
            paddle.uniform([300, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_9249ffd395a72e11f278f7c9595aad4c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1a7601697bb1f5580fda8c480301e31d
    def get_inputs(self):
        return [
            paddle.uniform([1, 1248], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_144b362bd75a37b105002d9b267898d1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1a7601697bb1f5580fda8c480301e31d
    def get_inputs(self):
        return [
            paddle.uniform([171, 480], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_b9209b4d5f6a21378b5ee8ef73f3735c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1a7601697bb1f5580fda8c480301e31d
    def get_inputs(self):
        return [
            paddle.uniform([145, 36], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_6458ca7cd7e31d68bad50fd8a9d78276(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_720ad26af3b2f0ee434c1ffd071cd7a2
    def get_inputs(self):
        return [
            paddle.uniform([15, 25], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_6458ca7cd7e31d68bad50fd8a9d78276(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_720ad26af3b2f0ee434c1ffd071cd7a2
    def get_inputs(self):
        return [
            paddle.uniform([15, 25], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0, 1], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_5f2230f817255c99bc611f3174793cc5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c9886c1b2ee6b6b0892424aab86cff0c
    def get_inputs(self):
        return [
            paddle.uniform([1, 18, 128, 256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_5f2230f817255c99bc611f3174793cc5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c9886c1b2ee6b6b0892424aab86cff0c
    def get_inputs(self):
        return [
            paddle.uniform([1, 18, 128, 256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_6df4e34015e0cc96f9eeeb690597533d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e1101708cb01271c757da4a74320dff3
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1, 4725], dtype='int32'),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_7e774555769d8ba1a595ed8eff162766(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2b5fb95cdadf8dd4d13f7dbab493720a
    def get_inputs(self):
        return [
            paddle.uniform([2347], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_844c837c98299e7a0cb856f7e0072137(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7df75c742412e3266ee6985318445ce4
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 4725], dtype='int32'), 'bool'),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_9f00525ff2bb555ec05a8bcfe39de4e2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_84df27c152d68c4a892169d2025e6df8
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[2347, 4], dtype='int64'),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_9f00525ff2bb555ec05a8bcfe39de4e2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_84df27c152d68c4a892169d2025e6df8
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[2347, 4], dtype='int64'),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_29379cd5d3ce0b6053fc93fd550068cb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c9886c1b2ee6b6b0892424aab86cff0c
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 32, 64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_29379cd5d3ce0b6053fc93fd550068cb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c9886c1b2ee6b6b0892424aab86cff0c
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 32, 64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_0124c9f9b60daefa7a2c557ede101619(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e1101708cb01271c757da4a74320dff3
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1, 6069], dtype='int32'),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_b095832f50fd28ac006252c2451ee974(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2b5fb95cdadf8dd4d13f7dbab493720a
    def get_inputs(self):
        return [
            paddle.uniform([3109], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_6dc9821f9ff7d163324a9862668688ab(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7df75c742412e3266ee6985318445ce4
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 6069], dtype='int32'), 'bool'),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_34df86e19f883a826c1b5955ab869be2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_84df27c152d68c4a892169d2025e6df8
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[3109, 4], dtype='int64'),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_34df86e19f883a826c1b5955ab869be2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_84df27c152d68c4a892169d2025e6df8
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[3109, 4], dtype='int64'),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_eabdaaf0771fc547eae5ecf0add543f8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e1101708cb01271c757da4a74320dff3
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1, 7581], dtype='int32'),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_2b1388963f7782f8c009301ceac8fddd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2b5fb95cdadf8dd4d13f7dbab493720a
    def get_inputs(self):
        return [
            paddle.uniform([3813], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_f93f5edb700ce29a43036a1dacf8f35b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7df75c742412e3266ee6985318445ce4
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 7581], dtype='int32'), 'bool'),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_de87e4d71e98da05701cf80ff25c1322(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_84df27c152d68c4a892169d2025e6df8
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[3813, 4], dtype='int64'),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_de87e4d71e98da05701cf80ff25c1322(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_84df27c152d68c4a892169d2025e6df8
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[3813, 4], dtype='int64'),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_9a9990b0b26403fef2c4fa4fa5b7e9fc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c9886c1b2ee6b6b0892424aab86cff0c
    def get_inputs(self):
        return [
            paddle.uniform([1, 48, 128, 256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_9a9990b0b26403fef2c4fa4fa5b7e9fc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c9886c1b2ee6b6b0892424aab86cff0c
    def get_inputs(self):
        return [
            paddle.uniform([1, 48, 128, 256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_3623b2239306ae612d9ed3393d43503d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1a7601697bb1f5580fda8c480301e31d
    def get_inputs(self):
        return [
            paddle.uniform([1, 156], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_c146f06d5c7503f2bd38bf3030d5019c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c9886c1b2ee6b6b0892424aab86cff0c
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 64, 128], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_c146f06d5c7503f2bd38bf3030d5019c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c9886c1b2ee6b6b0892424aab86cff0c
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 64, 128], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_621a2cbc345df2edbdb8d7cae3294180(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1a7601697bb1f5580fda8c480301e31d
    def get_inputs(self):
        return [
            paddle.uniform([1, 872], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_2406af63716c1119acf95fd9d9c45bd5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_62ceace5eda7bd790997aa9cf6734f7e
    def get_inputs(self):
        return [
            paddle.uniform([10, 8, 49, 28, 28], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_f7f4b87c351bfb6d5c50b1cef1a19327(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1a7601697bb1f5580fda8c480301e31d
    def get_inputs(self):
        return [
            paddle.uniform([22, 480], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_611b5558b6ec73dd3b7cbc6e54313db1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1a7601697bb1f5580fda8c480301e31d
    def get_inputs(self):
        return [
            paddle.uniform([145, 480], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_55ee00ce17b504e5532131e9ff270383(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1a7601697bb1f5580fda8c480301e31d
    def get_inputs(self):
        return [
            paddle.uniform([2, 80], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_0adfc035ce6676e11e0a0d680589bafc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_11421ecd2ecbbe423944eb7eb8f42226
    def get_inputs(self):
        return [
            paddle.to_tensor([0.32936179637908936, 0.08513082563877106], dtype='float32').reshape([2]),
            paddle.to_tensor([1, 2], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_9a67747bf8f551915205a482f1145cff(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1a7601697bb1f5580fda8c480301e31d
    def get_inputs(self):
        return [
            paddle.uniform([171, 36], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_23f5e6631597ec97a9db6bc7d3c94942(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_62ceace5eda7bd790997aa9cf6734f7e
    def get_inputs(self):
        return [
            paddle.uniform([22, 16, 49, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_01a3ff3485bc922e25575312bff0f841(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1a7601697bb1f5580fda8c480301e31d
    def get_inputs(self):
        return [
            paddle.uniform([1, 120], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_b8fd754ef34ee88055b2da5213f97870(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_62ceace5eda7bd790997aa9cf6734f7e
    def get_inputs(self):
        return [
            paddle.uniform([10, 32, 49, 7, 7], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_8d0ca9bf8a6d146093621192778a3e6c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_729e4377ce6c4d2c5ec9cfcf417fbffc
    def get_inputs(self):
        return [
            paddle.to_tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype='int64').reshape([20]),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_a3a09783f4bb50d11968b866657b73da(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_729e4377ce6c4d2c5ec9cfcf417fbffc
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype='int64').reshape([20]),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_00f223a733aaae7978de850a92156613(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1a7601697bb1f5580fda8c480301e31d
    def get_inputs(self):
        return [
            paddle.uniform([1, 672], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_2406af63716c1119acf95fd9d9c45bd5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_62ceace5eda7bd790997aa9cf6734f7e
    def get_inputs(self):
        return [
            paddle.uniform([10, 8, 49, 28, 28], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_1b2ef89f69602e34fe88118741570561(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e1101708cb01271c757da4a74320dff3
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1, 4116], dtype='int32'),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_686d73b6d6c737cf9cb363634d45bd37(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2b5fb95cdadf8dd4d13f7dbab493720a
    def get_inputs(self):
        return [
            paddle.uniform([2100], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_59ab69dfccef5aceec525e33e06ecee4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7df75c742412e3266ee6985318445ce4
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 4116], dtype='int32'), 'bool'),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_6882d36d971a13b6ee7191aeeaf99ebc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_84df27c152d68c4a892169d2025e6df8
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[2100, 4], dtype='int64'),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_6882d36d971a13b6ee7191aeeaf99ebc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_84df27c152d68c4a892169d2025e6df8
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[2100, 4], dtype='int64'),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_56d0f7a4639efedccd7f72c1eaa2c43b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5fbae16c52ec57d0c7462e48b41785a6
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1, 500], dtype='int64'),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_15485c1dfc6d7f868409cf0620c52376(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f3dacf69dd92f09b122a64ffb0d6262b
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1, 500], dtype='int32'),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_37b68b1d39c5622d1272a6649f8d5053(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_62ceace5eda7bd790997aa9cf6734f7e
    def get_inputs(self):
        return [
            paddle.uniform([22, 2, 9, 112, 112], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_f4e913ac460fa620d14a06317294f371(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_62ceace5eda7bd790997aa9cf6734f7e
    def get_inputs(self):
        return [
            paddle.uniform([22, 32, 49, 7, 7], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_1616f761412188d3de50c584a44759cc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cb1524500f57c783066b117e2c76f2c4
    def get_inputs(self):
        return [
            paddle.uniform([150, 256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_d76569b5b99cbfef13ca8c38b45211aa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e1101708cb01271c757da4a74320dff3
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1, 8400], dtype='int32'),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_adc4ba9b92bc0f6e99ef8b6d79502be5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2b5fb95cdadf8dd4d13f7dbab493720a
    def get_inputs(self):
        return [
            paddle.uniform([4231], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_b729cc6641dbd8e5394e8ab812ae05a7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7df75c742412e3266ee6985318445ce4
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 8400], dtype='int32'), 'bool'),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_ef029728c53ecaa9fae45584bdace686(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_84df27c152d68c4a892169d2025e6df8
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[4231, 4], dtype='int64'),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_ef029728c53ecaa9fae45584bdace686(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_84df27c152d68c4a892169d2025e6df8
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[4231, 4], dtype='int64'),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_e0a8af6f992ace4d178a3bfb9e6cab75(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1a7601697bb1f5580fda8c480301e31d
    def get_inputs(self):
        return [
            paddle.uniform([1, 624], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_23f5e6631597ec97a9db6bc7d3c94942(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_62ceace5eda7bd790997aa9cf6734f7e
    def get_inputs(self):
        return [
            paddle.uniform([22, 16, 49, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]




if __name__ == '__main__':
    unittest.main()