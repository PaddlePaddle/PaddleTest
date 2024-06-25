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



class PrimitiveOp_42d52191284e8ace46aa2a1dd0baaf6e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2, input_3):
        return paddle._C_ops.deformable_conv(input_0, input_1, input_2, input_3, [1, 1], [1, 1], [1, 1], 1, 1, 1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 258, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 18, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[128, 258, 3, 3], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 9, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_5343649dbd50cb60673fcfee5e095e88(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_42d52191284e8ace46aa2a1dd0baaf6e
    def get_inputs(self):
        return [
            paddle.uniform([1, 258, 24, 36], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 18, 24, 36], dtype='float32', min=0, max=0.5),
            paddle.uniform([128, 258, 3, 3], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 9, 24, 36], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_5dc9de9379539bad592f940c5af9075e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2, input_3):
        return paddle._C_ops.deformable_conv(input_0, input_1, input_2, input_3, [1, 1], [1, 1], [1, 1], 1, 1, 1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 256, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 18, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[256, 256, 3, 3], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 9, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_fdb4be96ada39046a6807f7fcde74a42(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5dc9de9379539bad592f940c5af9075e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 112, 160], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 18, 112, 160], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 9, 112, 160], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b4c7a5d4e99ed1b9c9b9bca59da77268(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5dc9de9379539bad592f940c5af9075e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 24, 24], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 18, 24, 24], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 9, 24, 24], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_7f1716e50025469e745196b46efba397(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2, input_3):
        return paddle._C_ops.deformable_conv(input_0, input_1, input_2, input_3, [1, 1], [1, 1], [1, 1], 1, 1, 1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 256, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 18, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[128, 256, 3, 3], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 9, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_8c9e483534937a343e5357a068bcaebf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7f1716e50025469e745196b46efba397
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 30, 50], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 18, 30, 50], dtype='float32', min=0, max=0.5),
            paddle.uniform([128, 256, 3, 3], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 9, 30, 50], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_f2f5b600c921594e95bff8d38aa1cb3d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2, input_3):
        return paddle._C_ops.deformable_conv(input_0, input_1, input_2, input_3, [1, 1], [1, 1], [1, 1], 1, 1, 1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 512, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 18, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[512, 512, 3, 3], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 9, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_36fabb14e74518b6f7e1f4dd0f60b382(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f2f5b600c921594e95bff8d38aa1cb3d
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 16, 16], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 18, 16, 16], dtype='float32', min=0, max=0.5),
            paddle.uniform([512, 512, 3, 3], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 9, 16, 16], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_61df3a8429ed8b6b3bee81be6fe13de2(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2, input_3):
        return paddle._C_ops.deformable_conv(input_0, input_1, input_2, input_3, [1, 1], [1, 1], [1, 1], 1, 1, 1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 258, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 18, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[512, 258, 3, 3], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 9, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_1fecd3c217d3da707d44cc6d8cb26e93(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_61df3a8429ed8b6b3bee81be6fe13de2
    def get_inputs(self):
        return [
            paddle.uniform([1, 258, 24, 24], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 18, 24, 24], dtype='float32', min=0, max=0.5),
            paddle.uniform([512, 258, 3, 3], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 9, 24, 24], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_196706e47347633ce52b540d9dbbe4eb(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2, input_3):
        return paddle._C_ops.deformable_conv(input_0, input_1, input_2, input_3, [1, 1], [1, 1], [1, 1], 1, 1, 1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 128, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 18, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[128, 128, 3, 3], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 9, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_3e03337f80a77349b3ea2f0a7f9956e8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_196706e47347633ce52b540d9dbbe4eb
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 48, 72], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 18, 48, 72], dtype='float32', min=0, max=0.5),
            paddle.uniform([128, 128, 3, 3], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 9, 48, 72], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_5ec8379aafe003a5f62b1719ed0a162d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_196706e47347633ce52b540d9dbbe4eb
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 60, 100], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 18, 60, 100], dtype='float32', min=0, max=0.5),
            paddle.uniform([128, 128, 3, 3], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 9, 60, 100], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_5400b5ed5b186918598b7c2be0200bb2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f2f5b600c921594e95bff8d38aa1cb3d
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 36, 36], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 18, 36, 36], dtype='float32', min=0, max=0.5),
            paddle.uniform([512, 512, 3, 3], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 9, 36, 36], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_0ecf74fd7637a2ed7fe71ed8e9d2751c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7f1716e50025469e745196b46efba397
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 60, 100], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 18, 60, 100], dtype='float32', min=0, max=0.5),
            paddle.uniform([128, 256, 3, 3], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 9, 60, 100], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d0e66ec9911ac4ff6559198590000a92(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5dc9de9379539bad592f940c5af9075e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 7, 10], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 18, 7, 10], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 9, 7, 10], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_40f12980b6db1677bfdda4f92480393c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f2f5b600c921594e95bff8d38aa1cb3d
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 24, 24], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 18, 24, 24], dtype='float32', min=0, max=0.5),
            paddle.uniform([512, 512, 3, 3], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 9, 24, 24], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_0f220d80f1770579964b1e9a439ce554(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5dc9de9379539bad592f940c5af9075e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 16, 16], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 18, 16, 16], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 9, 16, 16], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d76e301e8f1b7e77aae9e28021ef2f7b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f2f5b600c921594e95bff8d38aa1cb3d
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 12, 12], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 18, 12, 12], dtype='float32', min=0, max=0.5),
            paddle.uniform([512, 512, 3, 3], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 9, 12, 12], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6e10b9a788a33bcb9418ba1b06fbc311(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f2f5b600c921594e95bff8d38aa1cb3d
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 40, 40], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 18, 40, 40], dtype='float32', min=0, max=0.5),
            paddle.uniform([512, 512, 3, 3], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 9, 40, 40], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_2c9e7504cd45698dbff05db62bac06dd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5dc9de9379539bad592f940c5af9075e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 40, 40], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 18, 40, 40], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 9, 40, 40], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_5bb052448fd2cd505415a0ffd31cf918(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_61df3a8429ed8b6b3bee81be6fe13de2
    def get_inputs(self):
        return [
            paddle.uniform([1, 258, 40, 40], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 18, 40, 40], dtype='float32', min=0, max=0.5),
            paddle.uniform([512, 258, 3, 3], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 9, 40, 40], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_bfdcef1c7748ed579c4a19de73bb8717(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7f1716e50025469e745196b46efba397
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 48, 72], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 18, 48, 72], dtype='float32', min=0, max=0.5),
            paddle.uniform([128, 256, 3, 3], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 9, 48, 72], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_957a0a66cbd6a05bd9b7afa6c9c64e48(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_61df3a8429ed8b6b3bee81be6fe13de2
    def get_inputs(self):
        return [
            paddle.uniform([1, 258, 16, 16], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 18, 16, 16], dtype='float32', min=0, max=0.5),
            paddle.uniform([512, 258, 3, 3], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 9, 16, 16], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_0146059f425f9f83576cdfe5d7edc73c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_42d52191284e8ace46aa2a1dd0baaf6e
    def get_inputs(self):
        return [
            paddle.uniform([1, 258, 15, 25], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 18, 15, 25], dtype='float32', min=0, max=0.5),
            paddle.uniform([128, 258, 3, 3], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 9, 15, 25], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_75424a81ec8dc980a7265a7a944fb8fc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_196706e47347633ce52b540d9dbbe4eb
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 96, 144], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 18, 96, 144], dtype='float32', min=0, max=0.5),
            paddle.uniform([128, 128, 3, 3], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 9, 96, 144], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_aa4b2d1ad1a3b7f409c6663178d5a477(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7f1716e50025469e745196b46efba397
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 192, 288], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 18, 192, 288], dtype='float32', min=0, max=0.5),
            paddle.uniform([128, 256, 3, 3], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 9, 192, 288], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_82361ddc45ed4ab28f94c7638b98ca73(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7f1716e50025469e745196b46efba397
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 96, 144], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 18, 96, 144], dtype='float32', min=0, max=0.5),
            paddle.uniform([128, 256, 3, 3], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 9, 96, 144], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9ccb9aa156ff13eb4b681298a56b967b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_61df3a8429ed8b6b3bee81be6fe13de2
    def get_inputs(self):
        return [
            paddle.uniform([1, 258, 12, 12], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 18, 12, 12], dtype='float32', min=0, max=0.5),
            paddle.uniform([512, 258, 3, 3], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 9, 12, 12], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_55b65cacc158e2b9932ac83f040e2133(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5dc9de9379539bad592f940c5af9075e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 28, 40], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 18, 28, 40], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 9, 28, 40], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_cff1430d0ad7179d29ae71d072761d9a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7f1716e50025469e745196b46efba397
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 120, 200], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 18, 120, 200], dtype='float32', min=0, max=0.5),
            paddle.uniform([128, 256, 3, 3], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 9, 120, 200], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_4a2da7ae731c4c6d607de7b477bae897(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2, input_3):
        return paddle._C_ops.deformable_conv(input_0, input_1, input_2, input_3, [1, 1], [0, 0], [1, 1], 1, 1, 1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 128, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 2, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[128, 128, 1, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 1, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_f4c91763e66cceef84a575bf6db7ad2c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4a2da7ae731c4c6d607de7b477bae897
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 120, 200], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 2, 120, 200], dtype='float32', min=0, max=0.5),
            paddle.uniform([128, 128, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 1, 120, 200], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9ac1d9e29e2e19d30a4bbc7046a860c3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_61df3a8429ed8b6b3bee81be6fe13de2
    def get_inputs(self):
        return [
            paddle.uniform([1, 258, 36, 36], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 18, 36, 36], dtype='float32', min=0, max=0.5),
            paddle.uniform([512, 258, 3, 3], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 9, 36, 36], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_8b2fed5de63881e36b769dc2b228ac30(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5dc9de9379539bad592f940c5af9075e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 12, 12], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 18, 12, 12], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 9, 12, 12], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ec966c52fdf370ec37417f7ba76378cc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5dc9de9379539bad592f940c5af9075e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 56, 80], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 18, 56, 80], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 9, 56, 80], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_02334a0523974827d77502ccf15e7feb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5dc9de9379539bad592f940c5af9075e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 36, 36], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 18, 36, 36], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 9, 36, 36], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_7c4bb278026f11cbe8cf71d3fab23b93(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_196706e47347633ce52b540d9dbbe4eb
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 30, 50], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 18, 30, 50], dtype='float32', min=0, max=0.5),
            paddle.uniform([128, 128, 3, 3], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 9, 30, 50], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_fbd56fd099bb36c115834d8bc5f1ed29(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5dc9de9379539bad592f940c5af9075e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 14, 20], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 18, 14, 20], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 9, 14, 20], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_95119efa7bb6ef0d5805c6a6c1b829f5(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2, input_3):
        return paddle._C_ops.deformable_conv(input_0, input_1, input_2, input_3, [1, 1], [0, 0], [1, 1], 1, 1, 1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 128, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 2, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[256, 128, 1, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 1, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_8e47d08e1c07402bc6b41edaa0f9272d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_95119efa7bb6ef0d5805c6a6c1b829f5
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 192, 288], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 2, 192, 288], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 128, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 1, 192, 288], dtype='float32', min=0, max=0.5),
        ]




if __name__ == '__main__':
    unittest.main()