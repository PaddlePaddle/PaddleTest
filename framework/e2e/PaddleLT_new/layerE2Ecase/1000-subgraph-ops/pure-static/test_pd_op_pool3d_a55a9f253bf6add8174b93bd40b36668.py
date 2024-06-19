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



class PrimitiveOp_7cb5fb5dada5dcd6f71f7b32b44c7df4(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.pool3d(input_0, [2, 1, 1], [2, 1, 1], [0, 0, 0], False, True, 'NCDHW', 'avg', False, False, 'EXPLICIT')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 1, 192, 32, 64], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_d9a9a51b4b113ca1d90eb7e35d281e81(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7cb5fb5dada5dcd6f71f7b32b44c7df4
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 192, 32, 64], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_d9a9a51b4b113ca1d90eb7e35d281e81(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7cb5fb5dada5dcd6f71f7b32b44c7df4
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 192, 32, 64], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_9d195033b48f1587a9474726aeba9dae(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.pool3d(input_0, [2, 1, 1], [2, 1, 1], [0, 0, 0], False, True, 'NCDHW', 'avg', False, False, 'EXPLICIT')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 1, 96, 64, 128], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_5f9435d2bdf868a14a7eaf77f53fc0fb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9d195033b48f1587a9474726aeba9dae
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 96, 64, 128], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_5f9435d2bdf868a14a7eaf77f53fc0fb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9d195033b48f1587a9474726aeba9dae
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 96, 64, 128], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_3d1fbdbc78075f6c2549ee638ddc72bc(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.pool3d(input_0, [2, 1, 1], [2, 1, 1], [0, 0, 0], False, True, 'NCDHW', 'avg', False, False, 'EXPLICIT')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 1, 36, 64, 128], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_8761ddd3dffa140065b9702a5b3cae1c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3d1fbdbc78075f6c2549ee638ddc72bc
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 36, 64, 128], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_8761ddd3dffa140065b9702a5b3cae1c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3d1fbdbc78075f6c2549ee638ddc72bc
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 36, 64, 128], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_c080d641c8733763619e2164b3152c63(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.pool3d(input_0, [2, 1, 1], [2, 1, 1], [0, 0, 0], False, True, 'NCDHW', 'avg', False, False, 'EXPLICIT')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 1, 72, 32, 64], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_9b17942d46ae2bd2edbd3b114874e308(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c080d641c8733763619e2164b3152c63
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 72, 32, 64], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_9b17942d46ae2bd2edbd3b114874e308(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c080d641c8733763619e2164b3152c63
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 72, 32, 64], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_9a9ec2da26d3eeeae41d3ab8cbaabf76(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.pool3d(input_0, [2, 1, 1], [2, 1, 1], [0, 0, 0], False, True, 'NCDHW', 'avg', False, False, 'EXPLICIT')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 1, 18, 128, 256], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_ed91bf970893eaeb1c82c3bf9c008f83(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9a9ec2da26d3eeeae41d3ab8cbaabf76
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 18, 128, 256], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_ed91bf970893eaeb1c82c3bf9c008f83(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9a9ec2da26d3eeeae41d3ab8cbaabf76
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 18, 128, 256], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_0c621463959212f447675e2144538b6e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.pool3d(input_0, [2, 1, 1], [2, 1, 1], [0, 0, 0], False, True, 'NCDHW', 'avg', False, False, 'EXPLICIT')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 1, 48, 128, 256], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_51f754a7f2e1de716263746cb1a9cf7a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c621463959212f447675e2144538b6e
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 48, 128, 256], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_51f754a7f2e1de716263746cb1a9cf7a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c621463959212f447675e2144538b6e
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 48, 128, 256], dtype='float32', min=-0.5, max=0.5),
        ]




if __name__ == '__main__':
    unittest.main()