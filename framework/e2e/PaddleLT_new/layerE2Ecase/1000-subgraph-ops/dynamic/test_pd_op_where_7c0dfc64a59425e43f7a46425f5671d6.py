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



class PrimitiveOp_e9eb3a2ac209faa007e92f0d55682801(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        return paddle._C_ops.where(input_0, input_1, input_2)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None, None], dtype='bool'),
            paddle.static.InputSpec(shape=[None, None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None, None, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_b7169c668843c74cf4a88c5acd42d564(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e9eb3a2ac209faa007e92f0d55682801
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 3, 23, 23, 1], dtype='int32'), 'bool'),
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_b7169c668843c74cf4a88c5acd42d564(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e9eb3a2ac209faa007e92f0d55682801
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 3, 23, 23, 1], dtype='int32'), 'bool'),
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_a909bbc32f077ec2d4127bd257fc270a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        return paddle._C_ops.where(input_0, input_1, input_2)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='bool'),
            paddle.static.InputSpec(shape=[None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_e6e59cb28c787af276b430ac23d020b8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a909bbc32f077ec2d4127bd257fc270a
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=2, shape=[15200, 80], dtype='int32'), 'bool'),
            paddle.uniform([15200, 80], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([15200, 80], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_e6e59cb28c787af276b430ac23d020b8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a909bbc32f077ec2d4127bd257fc270a
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=2, shape=[15200, 80], dtype='int32'), 'bool'),
            paddle.uniform([15200, 80], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([15200, 80], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_57820c46cda365fef100b65051df58cb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a909bbc32f077ec2d4127bd257fc270a
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=2, shape=[64, 5], dtype='int32'), 'bool'),
            paddle.uniform([64, 5], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([64, 5], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_e0369778264a984c3cc0c7e0b270a4c2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e9eb3a2ac209faa007e92f0d55682801
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 3, 76, 76, 1], dtype='int32'), 'bool'),
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_e0369778264a984c3cc0c7e0b270a4c2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e9eb3a2ac209faa007e92f0d55682801
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 3, 76, 76, 1], dtype='int32'), 'bool'),
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_d5b4b9d5707570cc6ed46353b867329a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a909bbc32f077ec2d4127bd257fc270a
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=2, shape=[3800, 80], dtype='int32'), 'bool'),
            paddle.uniform([3800, 80], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([3800, 80], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_d5b4b9d5707570cc6ed46353b867329a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a909bbc32f077ec2d4127bd257fc270a
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=2, shape=[3800, 80], dtype='int32'), 'bool'),
            paddle.uniform([3800, 80], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([3800, 80], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_956822fff182946e9395956c772a00d1(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        return paddle._C_ops.where(input_0, input_1, input_2)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None], dtype='bool'),
            paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_079f69b86096f9524c8246e7a532d347(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_956822fff182946e9395956c772a00d1
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 21504, 2], dtype='int32'), 'bool'),
            paddle.uniform([1, 21504, 2], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 21504, 2], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_d5b4b9d5707570cc6ed46353b867329a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a909bbc32f077ec2d4127bd257fc270a
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=2, shape=[3800, 80], dtype='int32'), 'bool'),
            paddle.uniform([3800, 80], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([3800, 80], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_d5b4b9d5707570cc6ed46353b867329a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a909bbc32f077ec2d4127bd257fc270a
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=2, shape=[3800, 80], dtype='int32'), 'bool'),
            paddle.uniform([3800, 80], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([3800, 80], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_31431c4a677c1ae0c7c429a9aefa568c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        return paddle._C_ops.where(input_0, input_1, input_2)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='bool'),
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_e945bef1836f216a6eeb1b9bf7e50929(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_31431c4a677c1ae0c7c429a9aefa568c
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 2, 128, 128], dtype='int32'), 'bool'),
            paddle.uniform([1, 2, 128, 128], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 2, 128, 128], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_83412befd12debea55e2ef3c2bb4e52f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a909bbc32f077ec2d4127bd257fc270a
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=2, shape=[2204, 80], dtype='int32'), 'bool'),
            paddle.uniform([2204, 80], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([2204, 80], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_83412befd12debea55e2ef3c2bb4e52f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a909bbc32f077ec2d4127bd257fc270a
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=2, shape=[2204, 80], dtype='int32'), 'bool'),
            paddle.uniform([2204, 80], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([2204, 80], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_a8874d0d988035ae5e7e45adb3306350(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e9eb3a2ac209faa007e92f0d55682801
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 3, 21, 21, 1], dtype='int32'), 'bool'),
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_a8874d0d988035ae5e7e45adb3306350(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e9eb3a2ac209faa007e92f0d55682801
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 3, 21, 21, 1], dtype='int32'), 'bool'),
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_968ca60b3235334744979167198f0f64(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        return paddle._C_ops.where(input_0, input_1, input_2)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None], dtype='bool'),
            paddle.static.InputSpec(shape=[None], dtype='float32'),
            paddle.static.InputSpec(shape=[None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_1bc4436f181d758bea919a3103f79ba3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_968ca60b3235334744979167198f0f64
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=2, shape=[256], dtype='int32'), 'bool'),
            paddle.uniform([256], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([256], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_1bc4436f181d758bea919a3103f79ba3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_968ca60b3235334744979167198f0f64
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=2, shape=[256], dtype='int32'), 'bool'),
            paddle.uniform([256], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([256], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_dbd13ef11b3201883cda1bb26e45b0de(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a909bbc32f077ec2d4127bd257fc270a
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=2, shape=[950, 80], dtype='int32'), 'bool'),
            paddle.uniform([950, 80], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([950, 80], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_dbd13ef11b3201883cda1bb26e45b0de(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a909bbc32f077ec2d4127bd257fc270a
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=2, shape=[950, 80], dtype='int32'), 'bool'),
            paddle.uniform([950, 80], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([950, 80], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_7e076e9aae9926164ce4524c215598ea(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e9eb3a2ac209faa007e92f0d55682801
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 3, 84, 84, 1], dtype='int32'), 'bool'),
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_7e076e9aae9926164ce4524c215598ea(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e9eb3a2ac209faa007e92f0d55682801
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 3, 84, 84, 1], dtype='int32'), 'bool'),
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_5a4685c1e2dd50121500d61dcea6431d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_968ca60b3235334744979167198f0f64
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=2, shape=[4875], dtype='int32'), 'bool'),
            paddle.uniform([4875], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([4875], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_7ca5cee34c2b1cacce4d9a62ba8f08d6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_31431c4a677c1ae0c7c429a9aefa568c
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 2, 16, 16], dtype='int32'), 'bool'),
            paddle.uniform([1, 2, 16, 16], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 2, 16, 16], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_dd56d2dfcd01257d6fa26d2619932184(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a909bbc32f077ec2d4127bd257fc270a
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=2, shape=[8816, 80], dtype='int32'), 'bool'),
            paddle.uniform([8816, 80], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([8816, 80], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_dd56d2dfcd01257d6fa26d2619932184(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a909bbc32f077ec2d4127bd257fc270a
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=2, shape=[8816, 80], dtype='int32'), 'bool'),
            paddle.uniform([8816, 80], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([8816, 80], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_94236355c8b3d4908a2757c5121578cd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_31431c4a677c1ae0c7c429a9aefa568c
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 2, 64, 64], dtype='int32'), 'bool'),
            paddle.uniform([1, 2, 64, 64], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 2, 64, 64], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_287817d7c67f0ef537797db99125ed12(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_31431c4a677c1ae0c7c429a9aefa568c
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 2, 32, 32], dtype='int32'), 'bool'),
            paddle.uniform([1, 2, 32, 32], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 2, 32, 32], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_aca69b87b6d16108f387b7df6017bbb4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e9eb3a2ac209faa007e92f0d55682801
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 3, 42, 42, 1], dtype='int32'), 'bool'),
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_aca69b87b6d16108f387b7df6017bbb4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e9eb3a2ac209faa007e92f0d55682801
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 3, 42, 42, 1], dtype='int32'), 'bool'),
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_3ff545c765e98bcae0040c0a8564d76a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e9eb3a2ac209faa007e92f0d55682801
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 3, 92, 92, 1], dtype='int32'), 'bool'),
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_3ff545c765e98bcae0040c0a8564d76a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e9eb3a2ac209faa007e92f0d55682801
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 3, 92, 92, 1], dtype='int32'), 'bool'),
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_e1462287dda097e140bb530b82700852(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a909bbc32f077ec2d4127bd257fc270a
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=2, shape=[150, 80], dtype='int32'), 'bool'),
            paddle.uniform([150, 80], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([150, 80], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_e1462287dda097e140bb530b82700852(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a909bbc32f077ec2d4127bd257fc270a
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=2, shape=[150, 80], dtype='int32'), 'bool'),
            paddle.uniform([150, 80], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([150, 80], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_578205e7ecf0ab0058e5566e24111e02(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a909bbc32f077ec2d4127bd257fc270a
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=2, shape=[70, 80], dtype='int32'), 'bool'),
            paddle.uniform([70, 80], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([70, 80], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_578205e7ecf0ab0058e5566e24111e02(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a909bbc32f077ec2d4127bd257fc270a
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=2, shape=[70, 80], dtype='int32'), 'bool'),
            paddle.uniform([70, 80], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([70, 80], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_11cbc3b876c0190e0997e114a4b93756(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e9eb3a2ac209faa007e92f0d55682801
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 3, 44, 44, 1], dtype='int32'), 'bool'),
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_11cbc3b876c0190e0997e114a4b93756(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e9eb3a2ac209faa007e92f0d55682801
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 3, 44, 44, 1], dtype='int32'), 'bool'),
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_42246960638dbc869e82197de88c60bf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e9eb3a2ac209faa007e92f0d55682801
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 3, 11, 11, 1], dtype='int32'), 'bool'),
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_42246960638dbc869e82197de88c60bf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e9eb3a2ac209faa007e92f0d55682801
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 3, 11, 11, 1], dtype='int32'), 'bool'),
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_70ff17e048261b6b7964274b6ce63b82(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a909bbc32f077ec2d4127bd257fc270a
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=2, shape=[256, 5], dtype='int32'), 'bool'),
            paddle.uniform([256, 5], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([256, 5], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_7cce25297d24a719c0e0f4e4483876b6(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        return paddle._C_ops.where(input_0, input_1, input_2)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None], dtype='bool'),
            paddle.static.InputSpec(shape=[None], dtype='int32'),
            paddle.static.InputSpec(shape=[None], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_4d4c223ed0847b71c31b2e0789d5b370(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7cce25297d24a719c0e0f4e4483876b6
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=2, shape=[1002], dtype='int32'), 'bool'),
            paddle.randint(low=0, high=2, shape=[1002], dtype='int32'),
            paddle.randint(low=0, high=2, shape=[1002], dtype='int32'),
        ]


class TestPrimitiveOp_4d4c223ed0847b71c31b2e0789d5b370(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7cce25297d24a719c0e0f4e4483876b6
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=2, shape=[1002], dtype='int32'), 'bool'),
            paddle.randint(low=0, high=2, shape=[1002], dtype='int32'),
            paddle.randint(low=0, high=2, shape=[1002], dtype='int32'),
        ]


class TestPrimitiveOp_452a9196fbb0a9833be963070f4ff722(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e9eb3a2ac209faa007e92f0d55682801
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 3, 19, 19, 1], dtype='int32'), 'bool'),
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_452a9196fbb0a9833be963070f4ff722(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e9eb3a2ac209faa007e92f0d55682801
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 3, 19, 19, 1], dtype='int32'), 'bool'),
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_1bc4436f181d758bea919a3103f79ba3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_968ca60b3235334744979167198f0f64
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=2, shape=[256], dtype='int32'), 'bool'),
            paddle.uniform([256], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([256], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_1bc4436f181d758bea919a3103f79ba3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_968ca60b3235334744979167198f0f64
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=2, shape=[256], dtype='int32'), 'bool'),
            paddle.uniform([256], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([256], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_efaa73bc196aef0ef3004a01004a8b42(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e9eb3a2ac209faa007e92f0d55682801
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 3, 12, 12, 1], dtype='int32'), 'bool'),
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_efaa73bc196aef0ef3004a01004a8b42(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e9eb3a2ac209faa007e92f0d55682801
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 3, 12, 12, 1], dtype='int32'), 'bool'),
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_5a5623082c5b1c742844b5b9556c5a6c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_968ca60b3235334744979167198f0f64
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=2, shape=[17571], dtype='int32'), 'bool'),
            paddle.uniform([17571], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([17571], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_1c582e7960538c41f70eef7e3452cad9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e9eb3a2ac209faa007e92f0d55682801
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 3, 46, 46, 1], dtype='int32'), 'bool'),
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_1c582e7960538c41f70eef7e3452cad9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e9eb3a2ac209faa007e92f0d55682801
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 3, 46, 46, 1], dtype='int32'), 'bool'),
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_5ae5570f605a3afd084033a4e022b5e4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_31431c4a677c1ae0c7c429a9aefa568c
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 2, 8, 8], dtype='int32'), 'bool'),
            paddle.uniform([1, 2, 8, 8], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 2, 8, 8], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_cf8b658f2df9793c74a4e8acf2b150d2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a909bbc32f077ec2d4127bd257fc270a
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=2, shape=[16384, 5], dtype='int32'), 'bool'),
            paddle.uniform([16384, 5], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([16384, 5], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_7a624b6045fc279fae6e1745ea3c686b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e9eb3a2ac209faa007e92f0d55682801
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 3, 48, 48, 1], dtype='int32'), 'bool'),
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_7a624b6045fc279fae6e1745ea3c686b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e9eb3a2ac209faa007e92f0d55682801
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 3, 48, 48, 1], dtype='int32'), 'bool'),
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_47596c6faa249e6b2a76de0d00a670ef(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7cce25297d24a719c0e0f4e4483876b6
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=2, shape=[1027], dtype='int32'), 'bool'),
            paddle.randint(low=0, high=2, shape=[1027], dtype='int32'),
            paddle.randint(low=0, high=2, shape=[1027], dtype='int32'),
        ]


class TestPrimitiveOp_47596c6faa249e6b2a76de0d00a670ef(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7cce25297d24a719c0e0f4e4483876b6
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=2, shape=[1027], dtype='int32'), 'bool'),
            paddle.randint(low=0, high=2, shape=[1027], dtype='int32'),
            paddle.randint(low=0, high=2, shape=[1027], dtype='int32'),
        ]


class TestPrimitiveOp_c5e4663d6ad7f4f0657021ecd793461e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e9eb3a2ac209faa007e92f0d55682801
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 3, 24, 24, 1], dtype='int32'), 'bool'),
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_c5e4663d6ad7f4f0657021ecd793461e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e9eb3a2ac209faa007e92f0d55682801
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 3, 24, 24, 1], dtype='int32'), 'bool'),
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_d3ff2109f833ffbb0dfa976ae4333f5f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e9eb3a2ac209faa007e92f0d55682801
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 3, 38, 38, 1], dtype='int32'), 'bool'),
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_d3ff2109f833ffbb0dfa976ae4333f5f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e9eb3a2ac209faa007e92f0d55682801
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 3, 38, 38, 1], dtype='int32'), 'bool'),
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_b8853705b2b09d524e3814c2c9808b61(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a909bbc32f077ec2d4127bd257fc270a
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=2, shape=[551, 80], dtype='int32'), 'bool'),
            paddle.uniform([551, 80], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([551, 80], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_b8853705b2b09d524e3814c2c9808b61(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a909bbc32f077ec2d4127bd257fc270a
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=2, shape=[551, 80], dtype='int32'), 'bool'),
            paddle.uniform([551, 80], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([551, 80], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_e1573b1d78908a3ee07b0eb7eb89e707(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a909bbc32f077ec2d4127bd257fc270a
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=2, shape=[4096, 5], dtype='int32'), 'bool'),
            paddle.uniform([4096, 5], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([4096, 5], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_908420e8571d11c808beb9d13837f565(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e9eb3a2ac209faa007e92f0d55682801
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 3, 22, 22, 1], dtype='int32'), 'bool'),
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_908420e8571d11c808beb9d13837f565(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e9eb3a2ac209faa007e92f0d55682801
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 3, 22, 22, 1], dtype='int32'), 'bool'),
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_cb59cbaf3e4c56b326e62a9d5353429e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a909bbc32f077ec2d4127bd257fc270a
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=2, shape=[1024, 5], dtype='int32'), 'bool'),
            paddle.uniform([1024, 5], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1024, 5], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_54ee56853757ef8c7f3bca780d359c34(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7cce25297d24a719c0e0f4e4483876b6
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=2, shape=[2002], dtype='int32'), 'bool'),
            paddle.randint(low=0, high=2, shape=[2002], dtype='int32'),
            paddle.randint(low=0, high=2, shape=[2002], dtype='int32'),
        ]


class TestPrimitiveOp_54ee56853757ef8c7f3bca780d359c34(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7cce25297d24a719c0e0f4e4483876b6
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=2, shape=[2002], dtype='int32'), 'bool'),
            paddle.randint(low=0, high=2, shape=[2002], dtype='int32'),
            paddle.randint(low=0, high=2, shape=[2002], dtype='int32'),
        ]


class TestPrimitiveOp_479e484340bc333fb646e2f1218efb9d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7cce25297d24a719c0e0f4e4483876b6
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=2, shape=[1021], dtype='int32'), 'bool'),
            paddle.randint(low=0, high=2, shape=[1021], dtype='int32'),
            paddle.randint(low=0, high=2, shape=[1021], dtype='int32'),
        ]


class TestPrimitiveOp_479e484340bc333fb646e2f1218efb9d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7cce25297d24a719c0e0f4e4483876b6
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=2, shape=[1021], dtype='int32'), 'bool'),
            paddle.randint(low=0, high=2, shape=[1021], dtype='int32'),
            paddle.randint(low=0, high=2, shape=[1021], dtype='int32'),
        ]


class TestPrimitiveOp_40945cb79ec35282b2f29dfbd4a10fb9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a909bbc32f077ec2d4127bd257fc270a
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=2, shape=[247, 80], dtype='int32'), 'bool'),
            paddle.uniform([247, 80], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([247, 80], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_40945cb79ec35282b2f29dfbd4a10fb9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a909bbc32f077ec2d4127bd257fc270a
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=2, shape=[247, 80], dtype='int32'), 'bool'),
            paddle.uniform([247, 80], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([247, 80], dtype='float32', min=-0.5, max=0.5),
        ]




if __name__ == '__main__':
    unittest.main()