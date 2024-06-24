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



class PrimitiveOp_6216417761447a763e01c05cfeb660e1(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.hardsigmoid(input_0, 0.166667, 0.5)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_0b62ee5265cb301e4db266f8b3b5ddd3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6216417761447a763e01c05cfeb660e1
    def get_inputs(self):
        return [
            paddle.uniform([1, 480, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_0f9f0397143b126638db875af3e5d918(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.hardsigmoid(input_0, 0.2, 0.5)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_068b453fc74e782590b67fb0d7996589(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0f9f0397143b126638db875af3e5d918
    def get_inputs(self):
        return [
            paddle.uniform([1, 576, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_08ef299fd6cf9a78198bf186cafc3f52(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6216417761447a763e01c05cfeb660e1
    def get_inputs(self):
        return [
            paddle.uniform([1, 48, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_83d2636d411866883317c5fb75bd715d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6216417761447a763e01c05cfeb660e1
    def get_inputs(self):
        return [
            paddle.uniform([1, 160, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_70131dd0d0d6692d77d409664029233e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6216417761447a763e01c05cfeb660e1
    def get_inputs(self):
        return [
            paddle.uniform([1, 768, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_385a894a18c1cc3edd5df81f20b3416b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0f9f0397143b126638db875af3e5d918
    def get_inputs(self):
        return [
            paddle.uniform([1, 672, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b316ee25021c2091b4c8f9ce2b84ee9a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0f9f0397143b126638db875af3e5d918
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_35711064ee9f022bfebbc5e6948d4f5b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6216417761447a763e01c05cfeb660e1
    def get_inputs(self):
        return [
            paddle.uniform([1, 72, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_8cc9fa799a72ad2659f8b7c0b96d0113(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6216417761447a763e01c05cfeb660e1
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_020d5d9a7e7f8067bb7a1dbd61e6a94f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6216417761447a763e01c05cfeb660e1
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f3cc0043d41cc463a4f74f1e7413e54f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0f9f0397143b126638db875af3e5d918
    def get_inputs(self):
        return [
            paddle.uniform([1, 960, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_12024c3216b59f7466bf2a59b50d8f78(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0f9f0397143b126638db875af3e5d918
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ffb130dfdad40d9f6fed36e69a6f49fa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0f9f0397143b126638db875af3e5d918
    def get_inputs(self):
        return [
            paddle.uniform([1, 72, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f3cc0043d41cc463a4f74f1e7413e54f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0f9f0397143b126638db875af3e5d918
    def get_inputs(self):
        return [
            paddle.uniform([1, 960, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9e56b60507d3860d983e1de55933d172(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6216417761447a763e01c05cfeb660e1
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_020d5d9a7e7f8067bb7a1dbd61e6a94f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6216417761447a763e01c05cfeb660e1
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6d8c542abc69c84d1185cdd82701aed7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6216417761447a763e01c05cfeb660e1
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6d8c542abc69c84d1185cdd82701aed7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6216417761447a763e01c05cfeb660e1
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9e56b60507d3860d983e1de55933d172(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6216417761447a763e01c05cfeb660e1
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9e56b60507d3860d983e1de55933d172(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6216417761447a763e01c05cfeb660e1
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9a4462f092f0159b067f28c337323c6a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0f9f0397143b126638db875af3e5d918
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[1.3983314037322998]], [[1.5215564966201782]], [[0.9678478240966797]], [[1.885491967201233]], [[1.4255620241165161]], [[1.4419389963150024]], [[1.694916009902954]], [[1.7674769163131714]], [[1.361435890197754]], [[2.443833351135254]], [[0.9858337640762329]], [[2.4931530952453613]], [[1.7480871677398682]], [[1.8051831722259521]], [[1.6133379936218262]], [[2.1826589107513428]]]], dtype='float32').reshape([1, 16, 1, 1]),
        ]


class TestPrimitiveOp_9e56b60507d3860d983e1de55933d172(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6216417761447a763e01c05cfeb660e1
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_01ca996e778b88ffc5d9da9ea3139502(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6216417761447a763e01c05cfeb660e1
    def get_inputs(self):
        return [
            paddle.uniform([1, 44, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_83d2636d411866883317c5fb75bd715d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6216417761447a763e01c05cfeb660e1
    def get_inputs(self):
        return [
            paddle.uniform([1, 160, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f3cc0043d41cc463a4f74f1e7413e54f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0f9f0397143b126638db875af3e5d918
    def get_inputs(self):
        return [
            paddle.uniform([1, 960, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_0b62ee5265cb301e4db266f8b3b5ddd3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6216417761447a763e01c05cfeb660e1
    def get_inputs(self):
        return [
            paddle.uniform([1, 480, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b316ee25021c2091b4c8f9ce2b84ee9a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0f9f0397143b126638db875af3e5d918
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_385a894a18c1cc3edd5df81f20b3416b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0f9f0397143b126638db875af3e5d918
    def get_inputs(self):
        return [
            paddle.uniform([1, 672, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_1a4f766b55b1a973bdbc908703d74c77(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6216417761447a763e01c05cfeb660e1
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_385a894a18c1cc3edd5df81f20b3416b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0f9f0397143b126638db875af3e5d918
    def get_inputs(self):
        return [
            paddle.uniform([1, 672, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_0ea7f39f9f2ccc977b1aa64b3c572cb6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6216417761447a763e01c05cfeb660e1
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_0b62ee5265cb301e4db266f8b3b5ddd3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6216417761447a763e01c05cfeb660e1
    def get_inputs(self):
        return [
            paddle.uniform([1, 480, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_8cc9fa799a72ad2659f8b7c0b96d0113(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6216417761447a763e01c05cfeb660e1
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ee69249acd2df8d7aa7382586650a6f0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6216417761447a763e01c05cfeb660e1
    def get_inputs(self):
        return [
            paddle.uniform([1, 320, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_7ba9dfa32389d27c047b22cbedb7de9f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6216417761447a763e01c05cfeb660e1
    def get_inputs(self):
        return [
            paddle.uniform([1, 100, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f3cc0043d41cc463a4f74f1e7413e54f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0f9f0397143b126638db875af3e5d918
    def get_inputs(self):
        return [
            paddle.uniform([1, 960, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_385a894a18c1cc3edd5df81f20b3416b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0f9f0397143b126638db875af3e5d918
    def get_inputs(self):
        return [
            paddle.uniform([1, 672, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f45eb11c7a63e7e18d7980d3d6538f62(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6216417761447a763e01c05cfeb660e1
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_020d5d9a7e7f8067bb7a1dbd61e6a94f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6216417761447a763e01c05cfeb660e1
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6cdcd91821054abe5eaf8e8ab9fbf36a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6216417761447a763e01c05cfeb660e1
    def get_inputs(self):
        return [
            paddle.uniform([1, 240, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_8ad96dfb93fec51de96ff4edfa426d27(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6216417761447a763e01c05cfeb660e1
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_385a894a18c1cc3edd5df81f20b3416b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0f9f0397143b126638db875af3e5d918
    def get_inputs(self):
        return [
            paddle.uniform([1, 672, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ee69249acd2df8d7aa7382586650a6f0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6216417761447a763e01c05cfeb660e1
    def get_inputs(self):
        return [
            paddle.uniform([1, 320, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6d8c542abc69c84d1185cdd82701aed7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6216417761447a763e01c05cfeb660e1
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d910fcadfe0263d4c0fa8bb34aa104f3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0f9f0397143b126638db875af3e5d918
    def get_inputs(self):
        return [
            paddle.uniform([1, 480, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_385a894a18c1cc3edd5df81f20b3416b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0f9f0397143b126638db875af3e5d918
    def get_inputs(self):
        return [
            paddle.uniform([1, 672, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_35711064ee9f022bfebbc5e6948d4f5b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6216417761447a763e01c05cfeb660e1
    def get_inputs(self):
        return [
            paddle.uniform([1, 72, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_7f0b7441e83ac342be4ba6292a8eb69e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6216417761447a763e01c05cfeb660e1
    def get_inputs(self):
        return [
            paddle.uniform([1, 576, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_70131dd0d0d6692d77d409664029233e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6216417761447a763e01c05cfeb660e1
    def get_inputs(self):
        return [
            paddle.uniform([1, 768, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_068b453fc74e782590b67fb0d7996589(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0f9f0397143b126638db875af3e5d918
    def get_inputs(self):
        return [
            paddle.uniform([1, 576, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_7f0b7441e83ac342be4ba6292a8eb69e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6216417761447a763e01c05cfeb660e1
    def get_inputs(self):
        return [
            paddle.uniform([1, 576, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_020d5d9a7e7f8067bb7a1dbd61e6a94f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6216417761447a763e01c05cfeb660e1
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_385a894a18c1cc3edd5df81f20b3416b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0f9f0397143b126638db875af3e5d918
    def get_inputs(self):
        return [
            paddle.uniform([1, 672, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6cdcd91821054abe5eaf8e8ab9fbf36a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6216417761447a763e01c05cfeb660e1
    def get_inputs(self):
        return [
            paddle.uniform([1, 240, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ffb130dfdad40d9f6fed36e69a6f49fa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0f9f0397143b126638db875af3e5d918
    def get_inputs(self):
        return [
            paddle.uniform([1, 72, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6d8c542abc69c84d1185cdd82701aed7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6216417761447a763e01c05cfeb660e1
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_35711064ee9f022bfebbc5e6948d4f5b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6216417761447a763e01c05cfeb660e1
    def get_inputs(self):
        return [
            paddle.uniform([1, 72, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_08ef299fd6cf9a78198bf186cafc3f52(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6216417761447a763e01c05cfeb660e1
    def get_inputs(self):
        return [
            paddle.uniform([1, 48, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6d8c542abc69c84d1185cdd82701aed7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6216417761447a763e01c05cfeb660e1
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9e56b60507d3860d983e1de55933d172(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6216417761447a763e01c05cfeb660e1
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ffb130dfdad40d9f6fed36e69a6f49fa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0f9f0397143b126638db875af3e5d918
    def get_inputs(self):
        return [
            paddle.uniform([1, 72, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_8cc9fa799a72ad2659f8b7c0b96d0113(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6216417761447a763e01c05cfeb660e1
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9e56b60507d3860d983e1de55933d172(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6216417761447a763e01c05cfeb660e1
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9e56b60507d3860d983e1de55933d172(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6216417761447a763e01c05cfeb660e1
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_0ea7f39f9f2ccc977b1aa64b3c572cb6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6216417761447a763e01c05cfeb660e1
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_bbb3f0e493b7c3235f66d6f6a9a5c062(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6216417761447a763e01c05cfeb660e1
    def get_inputs(self):
        return [
            paddle.uniform([1, 144, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6d8c542abc69c84d1185cdd82701aed7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6216417761447a763e01c05cfeb660e1
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_020d5d9a7e7f8067bb7a1dbd61e6a94f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6216417761447a763e01c05cfeb660e1
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_70131dd0d0d6692d77d409664029233e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6216417761447a763e01c05cfeb660e1
    def get_inputs(self):
        return [
            paddle.uniform([1, 768, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_385a894a18c1cc3edd5df81f20b3416b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0f9f0397143b126638db875af3e5d918
    def get_inputs(self):
        return [
            paddle.uniform([1, 672, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_acea99becdd87ad03d23440d5b4c59ba(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6216417761447a763e01c05cfeb660e1
    def get_inputs(self):
        return [
            paddle.uniform([1, 288, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_08ef299fd6cf9a78198bf186cafc3f52(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6216417761447a763e01c05cfeb660e1
    def get_inputs(self):
        return [
            paddle.uniform([1, 48, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_020d5d9a7e7f8067bb7a1dbd61e6a94f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6216417761447a763e01c05cfeb660e1
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ffb130dfdad40d9f6fed36e69a6f49fa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0f9f0397143b126638db875af3e5d918
    def get_inputs(self):
        return [
            paddle.uniform([1, 72, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_08a8a2068f240481896f14a81266e7da(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0f9f0397143b126638db875af3e5d918
    def get_inputs(self):
        return [
            paddle.uniform([1, 240, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_83d2636d411866883317c5fb75bd715d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6216417761447a763e01c05cfeb660e1
    def get_inputs(self):
        return [
            paddle.uniform([1, 160, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6d8c542abc69c84d1185cdd82701aed7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6216417761447a763e01c05cfeb660e1
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f779877090132feaee4a1cac7ad200b6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0f9f0397143b126638db875af3e5d918
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d910fcadfe0263d4c0fa8bb34aa104f3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0f9f0397143b126638db875af3e5d918
    def get_inputs(self):
        return [
            paddle.uniform([1, 480, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ffb130dfdad40d9f6fed36e69a6f49fa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0f9f0397143b126638db875af3e5d918
    def get_inputs(self):
        return [
            paddle.uniform([1, 72, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_08ef299fd6cf9a78198bf186cafc3f52(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6216417761447a763e01c05cfeb660e1
    def get_inputs(self):
        return [
            paddle.uniform([1, 48, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6d8c542abc69c84d1185cdd82701aed7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6216417761447a763e01c05cfeb660e1
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_23ac52b2b22954f94b613a22c382dd9d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0f9f0397143b126638db875af3e5d918
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[2.475816249847412]], [[1.9965624809265137]], [[2.1500654220581055]], [[1.9056901931762695]], [[2.2067959308624268]], [[2.5478224754333496]], [[2.389434814453125]], [[2.1113216876983643]], [[2.5377626419067383]], [[1.7516134977340698]], [[3.0810773372650146]], [[1.8836690187454224]], [[2.343853235244751]], [[2.3030426502227783]], [[2.411494016647339]], [[2.110084295272827]]]], dtype='float32').reshape([1, 16, 1, 1]),
        ]


class TestPrimitiveOp_9e56b60507d3860d983e1de55933d172(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6216417761447a763e01c05cfeb660e1
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_08a8a2068f240481896f14a81266e7da(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0f9f0397143b126638db875af3e5d918
    def get_inputs(self):
        return [
            paddle.uniform([1, 240, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f45eb11c7a63e7e18d7980d3d6538f62(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6216417761447a763e01c05cfeb660e1
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_cee78f563f750c92fbfe738b6a9ead2d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6216417761447a763e01c05cfeb660e1
    def get_inputs(self):
        return [
            paddle.uniform([1, 400, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6cdcd91821054abe5eaf8e8ab9fbf36a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6216417761447a763e01c05cfeb660e1
    def get_inputs(self):
        return [
            paddle.uniform([1, 240, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_70f90422c86f9a5a313ac1d07e7dc633(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6216417761447a763e01c05cfeb660e1
    def get_inputs(self):
        return [
            paddle.uniform([1, 960, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_0b62ee5265cb301e4db266f8b3b5ddd3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6216417761447a763e01c05cfeb660e1
    def get_inputs(self):
        return [
            paddle.uniform([1, 480, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9e56b60507d3860d983e1de55933d172(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6216417761447a763e01c05cfeb660e1
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_385a894a18c1cc3edd5df81f20b3416b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0f9f0397143b126638db875af3e5d918
    def get_inputs(self):
        return [
            paddle.uniform([1, 672, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_8ad96dfb93fec51de96ff4edfa426d27(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6216417761447a763e01c05cfeb660e1
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_919a7f29a92a8f6bd046ed0607e69c7c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0f9f0397143b126638db875af3e5d918
    def get_inputs(self):
        return [
            paddle.uniform([1, 336, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6d8c542abc69c84d1185cdd82701aed7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6216417761447a763e01c05cfeb660e1
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_08ef299fd6cf9a78198bf186cafc3f52(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6216417761447a763e01c05cfeb660e1
    def get_inputs(self):
        return [
            paddle.uniform([1, 48, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_8ad96dfb93fec51de96ff4edfa426d27(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6216417761447a763e01c05cfeb660e1
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_01ca996e778b88ffc5d9da9ea3139502(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6216417761447a763e01c05cfeb660e1
    def get_inputs(self):
        return [
            paddle.uniform([1, 44, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d910fcadfe0263d4c0fa8bb34aa104f3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0f9f0397143b126638db875af3e5d918
    def get_inputs(self):
        return [
            paddle.uniform([1, 480, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_cee78f563f750c92fbfe738b6a9ead2d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6216417761447a763e01c05cfeb660e1
    def get_inputs(self):
        return [
            paddle.uniform([1, 400, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3d16c5ca5ba90c48de12fe60e7a5b098(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6216417761447a763e01c05cfeb660e1
    def get_inputs(self):
        return [
            paddle.uniform([1, 56, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_020d5d9a7e7f8067bb7a1dbd61e6a94f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6216417761447a763e01c05cfeb660e1
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_0b62ee5265cb301e4db266f8b3b5ddd3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6216417761447a763e01c05cfeb660e1
    def get_inputs(self):
        return [
            paddle.uniform([1, 480, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_bbb3f0e493b7c3235f66d6f6a9a5c062(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6216417761447a763e01c05cfeb660e1
    def get_inputs(self):
        return [
            paddle.uniform([1, 144, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_83d2636d411866883317c5fb75bd715d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6216417761447a763e01c05cfeb660e1
    def get_inputs(self):
        return [
            paddle.uniform([1, 160, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_8ad96dfb93fec51de96ff4edfa426d27(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6216417761447a763e01c05cfeb660e1
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_020d5d9a7e7f8067bb7a1dbd61e6a94f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6216417761447a763e01c05cfeb660e1
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_387e8d75d5c3e74b581d3a3ae4eb43c8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0f9f0397143b126638db875af3e5d918
    def get_inputs(self):
        return [
            paddle.uniform([2, 96, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_387e8d75d5c3e74b581d3a3ae4eb43c8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0f9f0397143b126638db875af3e5d918
    def get_inputs(self):
        return [
            paddle.uniform([2, 96, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_387e8d75d5c3e74b581d3a3ae4eb43c8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0f9f0397143b126638db875af3e5d918
    def get_inputs(self):
        return [
            paddle.uniform([2, 96, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_387e8d75d5c3e74b581d3a3ae4eb43c8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0f9f0397143b126638db875af3e5d918
    def get_inputs(self):
        return [
            paddle.uniform([2, 96, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_fae8135640fe442c9fa85599ed89a656(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0f9f0397143b126638db875af3e5d918
    def get_inputs(self):
        return [
            paddle.uniform([2, 24, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_fae8135640fe442c9fa85599ed89a656(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0f9f0397143b126638db875af3e5d918
    def get_inputs(self):
        return [
            paddle.uniform([2, 24, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_fae8135640fe442c9fa85599ed89a656(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0f9f0397143b126638db875af3e5d918
    def get_inputs(self):
        return [
            paddle.uniform([2, 24, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_fae8135640fe442c9fa85599ed89a656(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0f9f0397143b126638db875af3e5d918
    def get_inputs(self):
        return [
            paddle.uniform([2, 24, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d910fcadfe0263d4c0fa8bb34aa104f3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0f9f0397143b126638db875af3e5d918
    def get_inputs(self):
        return [
            paddle.uniform([1, 480, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9e56b60507d3860d983e1de55933d172(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6216417761447a763e01c05cfeb660e1
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b316ee25021c2091b4c8f9ce2b84ee9a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0f9f0397143b126638db875af3e5d918
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b316ee25021c2091b4c8f9ce2b84ee9a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0f9f0397143b126638db875af3e5d918
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b316ee25021c2091b4c8f9ce2b84ee9a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0f9f0397143b126638db875af3e5d918
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4c8710ef9749fbcffa404a6c272ad415(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6216417761447a763e01c05cfeb660e1
    def get_inputs(self):
        return [
            paddle.uniform([1, 200, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_08a8a2068f240481896f14a81266e7da(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0f9f0397143b126638db875af3e5d918
    def get_inputs(self):
        return [
            paddle.uniform([1, 240, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_020d5d9a7e7f8067bb7a1dbd61e6a94f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6216417761447a763e01c05cfeb660e1
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b316ee25021c2091b4c8f9ce2b84ee9a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0f9f0397143b126638db875af3e5d918
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_08ef299fd6cf9a78198bf186cafc3f52(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6216417761447a763e01c05cfeb660e1
    def get_inputs(self):
        return [
            paddle.uniform([1, 48, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_08ef299fd6cf9a78198bf186cafc3f52(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6216417761447a763e01c05cfeb660e1
    def get_inputs(self):
        return [
            paddle.uniform([1, 48, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_7ba9dfa32389d27c047b22cbedb7de9f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6216417761447a763e01c05cfeb660e1
    def get_inputs(self):
        return [
            paddle.uniform([1, 100, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6cdcd91821054abe5eaf8e8ab9fbf36a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6216417761447a763e01c05cfeb660e1
    def get_inputs(self):
        return [
            paddle.uniform([1, 240, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9f740a257a2f73a087205d1b71053ed7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0f9f0397143b126638db875af3e5d918
    def get_inputs(self):
        return [
            paddle.uniform([1, 288, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_020d5d9a7e7f8067bb7a1dbd61e6a94f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6216417761447a763e01c05cfeb660e1
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_70131dd0d0d6692d77d409664029233e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6216417761447a763e01c05cfeb660e1
    def get_inputs(self):
        return [
            paddle.uniform([1, 768, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_acea99becdd87ad03d23440d5b4c59ba(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6216417761447a763e01c05cfeb660e1
    def get_inputs(self):
        return [
            paddle.uniform([1, 288, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_08ef299fd6cf9a78198bf186cafc3f52(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6216417761447a763e01c05cfeb660e1
    def get_inputs(self):
        return [
            paddle.uniform([1, 48, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_35711064ee9f022bfebbc5e6948d4f5b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6216417761447a763e01c05cfeb660e1
    def get_inputs(self):
        return [
            paddle.uniform([1, 72, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_70131dd0d0d6692d77d409664029233e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6216417761447a763e01c05cfeb660e1
    def get_inputs(self):
        return [
            paddle.uniform([1, 768, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9e56b60507d3860d983e1de55933d172(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6216417761447a763e01c05cfeb660e1
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9e56b60507d3860d983e1de55933d172(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6216417761447a763e01c05cfeb660e1
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9e56b60507d3860d983e1de55933d172(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6216417761447a763e01c05cfeb660e1
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4c8710ef9749fbcffa404a6c272ad415(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6216417761447a763e01c05cfeb660e1
    def get_inputs(self):
        return [
            paddle.uniform([1, 200, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6d8c542abc69c84d1185cdd82701aed7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6216417761447a763e01c05cfeb660e1
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ee69249acd2df8d7aa7382586650a6f0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6216417761447a763e01c05cfeb660e1
    def get_inputs(self):
        return [
            paddle.uniform([1, 320, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6d8c542abc69c84d1185cdd82701aed7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6216417761447a763e01c05cfeb660e1
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d910fcadfe0263d4c0fa8bb34aa104f3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0f9f0397143b126638db875af3e5d918
    def get_inputs(self):
        return [
            paddle.uniform([1, 480, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_020d5d9a7e7f8067bb7a1dbd61e6a94f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6216417761447a763e01c05cfeb660e1
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6d8c542abc69c84d1185cdd82701aed7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6216417761447a763e01c05cfeb660e1
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_7474c62983fab672108db31beb918aba(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0f9f0397143b126638db875af3e5d918
    def get_inputs(self):
        return [
            paddle.uniform([1, 40, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_020d5d9a7e7f8067bb7a1dbd61e6a94f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6216417761447a763e01c05cfeb660e1
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d910fcadfe0263d4c0fa8bb34aa104f3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0f9f0397143b126638db875af3e5d918
    def get_inputs(self):
        return [
            paddle.uniform([1, 480, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ffb130dfdad40d9f6fed36e69a6f49fa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0f9f0397143b126638db875af3e5d918
    def get_inputs(self):
        return [
            paddle.uniform([1, 72, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_acea99becdd87ad03d23440d5b4c59ba(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6216417761447a763e01c05cfeb660e1
    def get_inputs(self):
        return [
            paddle.uniform([1, 288, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_385a894a18c1cc3edd5df81f20b3416b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0f9f0397143b126638db875af3e5d918
    def get_inputs(self):
        return [
            paddle.uniform([1, 672, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_385a894a18c1cc3edd5df81f20b3416b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0f9f0397143b126638db875af3e5d918
    def get_inputs(self):
        return [
            paddle.uniform([1, 672, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6cdcd91821054abe5eaf8e8ab9fbf36a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6216417761447a763e01c05cfeb660e1
    def get_inputs(self):
        return [
            paddle.uniform([1, 240, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b316ee25021c2091b4c8f9ce2b84ee9a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0f9f0397143b126638db875af3e5d918
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_83d2636d411866883317c5fb75bd715d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6216417761447a763e01c05cfeb660e1
    def get_inputs(self):
        return [
            paddle.uniform([1, 160, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_08ef299fd6cf9a78198bf186cafc3f52(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6216417761447a763e01c05cfeb660e1
    def get_inputs(self):
        return [
            paddle.uniform([1, 48, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_1a4f766b55b1a973bdbc908703d74c77(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6216417761447a763e01c05cfeb660e1
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_0ea7f39f9f2ccc977b1aa64b3c572cb6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6216417761447a763e01c05cfeb660e1
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3822f79efaf6fe68062d7a3dd7db88bf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0f9f0397143b126638db875af3e5d918
    def get_inputs(self):
        return [
            paddle.uniform([1, 144, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_919a7f29a92a8f6bd046ed0607e69c7c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0f9f0397143b126638db875af3e5d918
    def get_inputs(self):
        return [
            paddle.uniform([1, 336, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_bbb3f0e493b7c3235f66d6f6a9a5c062(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6216417761447a763e01c05cfeb660e1
    def get_inputs(self):
        return [
            paddle.uniform([1, 144, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_020d5d9a7e7f8067bb7a1dbd61e6a94f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6216417761447a763e01c05cfeb660e1
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3d16c5ca5ba90c48de12fe60e7a5b098(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6216417761447a763e01c05cfeb660e1
    def get_inputs(self):
        return [
            paddle.uniform([1, 56, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_7f0b7441e83ac342be4ba6292a8eb69e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6216417761447a763e01c05cfeb660e1
    def get_inputs(self):
        return [
            paddle.uniform([1, 576, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_8ad96dfb93fec51de96ff4edfa426d27(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6216417761447a763e01c05cfeb660e1
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ee69249acd2df8d7aa7382586650a6f0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6216417761447a763e01c05cfeb660e1
    def get_inputs(self):
        return [
            paddle.uniform([1, 320, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b316ee25021c2091b4c8f9ce2b84ee9a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0f9f0397143b126638db875af3e5d918
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f3cc0043d41cc463a4f74f1e7413e54f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0f9f0397143b126638db875af3e5d918
    def get_inputs(self):
        return [
            paddle.uniform([1, 960, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9e56b60507d3860d983e1de55933d172(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6216417761447a763e01c05cfeb660e1
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_83d2636d411866883317c5fb75bd715d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6216417761447a763e01c05cfeb660e1
    def get_inputs(self):
        return [
            paddle.uniform([1, 160, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3822f79efaf6fe68062d7a3dd7db88bf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0f9f0397143b126638db875af3e5d918
    def get_inputs(self):
        return [
            paddle.uniform([1, 144, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_35711064ee9f022bfebbc5e6948d4f5b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6216417761447a763e01c05cfeb660e1
    def get_inputs(self):
        return [
            paddle.uniform([1, 72, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_70f90422c86f9a5a313ac1d07e7dc633(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6216417761447a763e01c05cfeb660e1
    def get_inputs(self):
        return [
            paddle.uniform([1, 960, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9e56b60507d3860d983e1de55933d172(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6216417761447a763e01c05cfeb660e1
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_8cc9fa799a72ad2659f8b7c0b96d0113(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6216417761447a763e01c05cfeb660e1
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4c8710ef9749fbcffa404a6c272ad415(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6216417761447a763e01c05cfeb660e1
    def get_inputs(self):
        return [
            paddle.uniform([1, 200, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_cee78f563f750c92fbfe738b6a9ead2d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6216417761447a763e01c05cfeb660e1
    def get_inputs(self):
        return [
            paddle.uniform([1, 400, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6d8c542abc69c84d1185cdd82701aed7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6216417761447a763e01c05cfeb660e1
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6d8c542abc69c84d1185cdd82701aed7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6216417761447a763e01c05cfeb660e1
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_7ba9dfa32389d27c047b22cbedb7de9f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6216417761447a763e01c05cfeb660e1
    def get_inputs(self):
        return [
            paddle.uniform([1, 100, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_08ef299fd6cf9a78198bf186cafc3f52(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6216417761447a763e01c05cfeb660e1
    def get_inputs(self):
        return [
            paddle.uniform([1, 48, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d910fcadfe0263d4c0fa8bb34aa104f3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0f9f0397143b126638db875af3e5d918
    def get_inputs(self):
        return [
            paddle.uniform([1, 480, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_83d2636d411866883317c5fb75bd715d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6216417761447a763e01c05cfeb660e1
    def get_inputs(self):
        return [
            paddle.uniform([1, 160, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ee69249acd2df8d7aa7382586650a6f0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6216417761447a763e01c05cfeb660e1
    def get_inputs(self):
        return [
            paddle.uniform([1, 320, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_0ea7f39f9f2ccc977b1aa64b3c572cb6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6216417761447a763e01c05cfeb660e1
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_83d2636d411866883317c5fb75bd715d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6216417761447a763e01c05cfeb660e1
    def get_inputs(self):
        return [
            paddle.uniform([1, 160, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d910fcadfe0263d4c0fa8bb34aa104f3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0f9f0397143b126638db875af3e5d918
    def get_inputs(self):
        return [
            paddle.uniform([1, 480, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6d8c542abc69c84d1185cdd82701aed7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6216417761447a763e01c05cfeb660e1
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_12024c3216b59f7466bf2a59b50d8f78(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0f9f0397143b126638db875af3e5d918
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_12024c3216b59f7466bf2a59b50d8f78(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0f9f0397143b126638db875af3e5d918
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_12024c3216b59f7466bf2a59b50d8f78(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0f9f0397143b126638db875af3e5d918
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_12024c3216b59f7466bf2a59b50d8f78(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0f9f0397143b126638db875af3e5d918
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_0c15b8e2a14ba6f8267b76988588b2ec(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0f9f0397143b126638db875af3e5d918
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[45832.83984375]], [[64038.4609375]], [[23228.36328125]], [[54584.9296875]], [[60669.34765625]], [[28748.931640625]], [[50911.51171875]], [[66960.0]], [[51999.5859375]], [[76857.875]], [[48020.6640625]], [[60172.80859375]], [[35579.46875]], [[61028.27734375]], [[42722.62890625]], [[57977.69921875]], [[65027.125]], [[59155.9453125]], [[62545.33203125]], [[64820.42578125]], [[34388.28125]], [[51160.640625]], [[47434.3828125]], [[45792.71484375]]]], dtype='float32').reshape([1, 24, 1, 1]),
        ]


class TestPrimitiveOp_3bdcf71a835e69df91b1b4f4f2af8b58(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0f9f0397143b126638db875af3e5d918
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[60768.1875]], [[62214.98046875]], [[49159.09375]], [[62979.171875]], [[66467.3046875]], [[83702.5546875]], [[49493.2890625]], [[49485.5625]], [[28647.755859375]], [[75862.15625]], [[52264.71875]], [[79810.703125]], [[77782.328125]], [[62766.99609375]], [[39632.6171875]], [[73498.9296875]], [[67134.0703125]], [[81753.7265625]], [[80952.203125]], [[79211.4765625]], [[47392.93359375]], [[56408.375]], [[78073.546875]], [[50775.66015625]]]], dtype='float32').reshape([1, 24, 1, 1]),
        ]


class TestPrimitiveOp_6b92e9faf03f4ff57315cb513884b7a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0f9f0397143b126638db875af3e5d918
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[79112.703125]], [[60442.1953125]], [[66346.2421875]], [[86740.7109375]], [[50277.91015625]], [[50255.49609375]], [[65666.7890625]], [[62602.96484375]], [[47060.48828125]], [[47561.21875]], [[56227.015625]], [[48668.7734375]], [[82128.09375]], [[60948.24609375]], [[59137.5625]], [[66859.9140625]], [[37535.671875]], [[44502.79296875]], [[58906.64453125]], [[52158.390625]], [[53154.8125]], [[56914.64453125]], [[56234.2265625]], [[50044.859375]]]], dtype='float32').reshape([1, 24, 1, 1]),
        ]


class TestPrimitiveOp_0f8bf7da7442bf371ea3aa540646080e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0f9f0397143b126638db875af3e5d918
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[88840.890625]], [[49344.23828125]], [[29987.88671875]], [[66598.6484375]], [[55961.62109375]], [[53273.80078125]], [[56786.5078125]], [[49842.671875]], [[68749.25]], [[54096.890625]], [[65380.734375]], [[74652.828125]], [[73266.8203125]], [[55279.234375]], [[59964.33203125]], [[58059.984375]], [[66096.5546875]], [[38738.6171875]], [[70089.78125]], [[96187.3125]], [[44262.8125]], [[24172.4453125]], [[47855.0859375]], [[70403.1875]]]], dtype='float32').reshape([1, 24, 1, 1]),
        ]


class TestPrimitiveOp_f3cc0043d41cc463a4f74f1e7413e54f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0f9f0397143b126638db875af3e5d918
    def get_inputs(self):
        return [
            paddle.uniform([1, 960, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9f740a257a2f73a087205d1b71053ed7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0f9f0397143b126638db875af3e5d918
    def get_inputs(self):
        return [
            paddle.uniform([1, 288, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_12024c3216b59f7466bf2a59b50d8f78(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0f9f0397143b126638db875af3e5d918
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_35711064ee9f022bfebbc5e6948d4f5b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6216417761447a763e01c05cfeb660e1
    def get_inputs(self):
        return [
            paddle.uniform([1, 72, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_385a894a18c1cc3edd5df81f20b3416b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0f9f0397143b126638db875af3e5d918
    def get_inputs(self):
        return [
            paddle.uniform([1, 672, 1, 1], dtype='float32', min=0, max=0.5),
        ]




if __name__ == '__main__':
    unittest.main()